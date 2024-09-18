# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mllama model."""

from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
import math

from transformers.activations import ACT2FN
import torch.nn.functional as F
from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
)
from text_generation_server.layers.attention import Seqlen
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    SpeculativeHead,
    FastLinear,
)
from text_generation_server.utils.weights import DefaultWeightsLoader, UnquantizedWeight


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->MllamaVision
class MllamaVisionMLP(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.fc1", weights=weights, config=config, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(
            prefix=f"{prefix}.fc2", weights=weights, config=config, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MllamaVisionSdpaAttention(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads

        self.qkv_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_state)
        query, key, value = qkv.split(
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
            ],
            dim=1,
        )

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)
        return output


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, *, prefix, config, weights, is_gated: bool):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MllamaVisionSdpaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = MllamaVisionMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights
        )

        self.input_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=1e-05
        )
        self.post_attention_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=1e-05
        )

        # there used to be an if else here, no code path
        if is_gated:
            self.gate_attn = nn.Parameter(
                weights.get_tensor(f"{prefix}.gate_attn"), requires_grad=False
            )
            self.gate_ffn = nn.Parameter(
                weights.get_tensor(f"{prefix}.gate_attn"), requires_grad=False
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state, attn_weights = self.self_attn(
            hidden_state, attention_mask=attention_mask
        )
        gate_attn = 1 if not self.is_gated else self.gate_attn.tanh()
        hidden_state = residual + gate_attn * hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        gate_ffn = 1 if not self.is_gated else self.gate_ffn.tanh()
        hidden_state = residual + gate_ffn * hidden_state
        return hidden_state


class MllamaVisionEncoder(nn.Module):
    def __init__(self, *, prefix, config, weights, is_gated: bool, num_layers: int):
        super().__init__()
        self.config = config
        self.layers = [
            MllamaVisionEncoderLayer(
                prefix=f"{prefix}.layers.{i}",
                config=config,
                weights=weights,
                is_gated=is_gated,
            )
            for i in range(num_layers)
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
            )

            hidden_states = layer_outputs[0]

        return hidden_states


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id

        self.embedding = TensorParallelEmbedding(
            prefix=f"{prefix}.embedding", weights=weights
        )
        self.gate = nn.Parameter(
            weights.get_tensor(f"{prefix}.gate"), requires_grad=False
        )

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        # Always gated.
        embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        self.embedding = nn.Parameter(
            weights.get_tensor(f"{prefix}.embedding"), requires_grad=False
        )
        self.tile_embedding = TensorParallelEmbedding(
            prefix=f"{prefix}.tile_embedding", weights=weights
        )

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(
            1, 1, self.num_patches, self.hidden_size
        )

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


class MllamaVisionModel(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.in_channels = config.in_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )
        self.patch_embedding.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embedding.weight"), requires_grad=False
        )

        self.class_embedding = nn.Parameter(
            weights.get_tensor(f"{prefix}.class_embedding"), requires_grad=False
        )

        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(
            prefix=f"{prefix}.gated_positional_embedding",
            config=config,
            weights=weights,
        )

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            prefix=f"{prefix}.pre_tile_positional_embedding",
            config=config,
            weights=weights,
        )
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            prefix=f"{prefix}.post_tile_positional_embedding",
            config=config,
            weights=weights,
        )

        ## layer norms
        self.layernorm_pre = nn.LayerNorm.load(
            prefix=f"{prefix}.layernorm_pre",
            weights=weights,
            # torch default
            eps=1e-05,
        )
        self.layernorm_post = nn.LayerNorm.load(
            prefix=f"{prefix}.layernorm_post",
            weights=weights,
            # torch default
            eps=1e-05,
        )

        ## encoders
        self.transformer = MllamaVisionEncoder(
            prefix=f"{prefix}.transformer",
            config=config,
            weights=weights,
            is_gated=False,
            num_layers=config.num_hidden_layers,
        )
        self.global_transformer = MllamaVisionEncoder(
            prefix=f"{prefix}.global_transformer",
            config=config,
            weights=weights,
            is_gated=True,
            num_layers=config.num_global_layers,
        )

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = (
            pixel_values.shape
        )

        pixel_values = pixel_values.reshape(
            batch_size * num_concurrent_media * num_tiles, num_channels, height, width
        )
        aspect_ratio_ids = aspect_ratio_ids.reshape(
            batch_size * num_concurrent_media, -1
        )

        # patch embedding
        patch_embeds = self.patch_embedding(pixel_values.to(self.dtype).to(self.device))
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, -1, dim
        )
        hidden_state = self.pre_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )

        # apply cls token
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media * num_tiles, num_patches, dim
        )
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # apply position embeddings
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches, dim
        )
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        # apply encoder
        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (
            0,
            0,
            0,
            num_padding_patches,
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(
                batch_size * num_concurrent_media, -1
            )
            attention_mask = _prepare_aspect_ratio_attention_mask(
                aspect_ratio_mask=attention_mask,
                num_patches=self.num_patches,
                target_length=hidden_state.shape[2],
                dtype=self.dtype,
            )

        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_state, all_intermediate_hidden_states = output[0], output[1]
        intermediate_hidden_states = [
            hidden_state
            for idx, hidden_state in enumerate(all_intermediate_hidden_states)
            if idx in self.intermediate_layers_indices
        ]
        intermediate_hidden_states = torch.stack(intermediate_hidden_states, dim=-1)

        # apply global encoder
        hidden_state = self.layernorm_post(hidden_state)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = self.post_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches),
            dim,
        )
        hidden_state = self.global_transformer(
            hidden_state, attention_mask=attention_mask
        )[0]
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = hidden_state[:, :, :slice_index]

        # adding intermediate layer outputs
        hidden_state = hidden_state.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, dim
        )
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            -1,
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)
        return hidden_state


class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.qkv_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

        self.q_norm = FastRMSNorm.load(
            prefix=f"{prefix}.q_norm", weights=weights, eps=config.rms_norm_eps
        )
        self.k_norm = FastRMSNorm.load(
            prefix=f"{prefix}.k_norm", weights=weights, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            key_states = self.k_norm(key_states)
            if past_key_value is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )
        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.gemma2.modeling_gemma2.Gemma2MLP with Gemma2->MllamaText
class MllamaTextMLP(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        gate_up_states = self.gate_up_proj(x)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(self, *, prefix, config, weights) -> None:
        super().__init__()
        self.cross_attn = MllamaTextCrossAttention(
            prefix=f"{prefix}.cross_attn", config=config, weights=weights
        )

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.cross_attn_attn_gate = torch.nn.Parameter(
            weights.get_tensor(f"{prefix}.cross_attn_attn_gate"), requires_grad=False
        )

        self.mlp = MllamaTextMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.cross_attn_mlp_gate = torch.nn.Parameter(
            weights.get_tensor(f"{prefix}.cross_attn_mlp_gate"), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class MllamaTextSelfAttention(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.qkv_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with LlamaDecoder->MllamaSelfAttentionDecoder, Llama->MllamaText, LLAMA->MLLAMA_TEXT
class MllamaSelfAttentionDecoderLayer(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MllamaTextSelfAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )

        self.mlp = MllamaTextMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MllamaTextModel(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens", weights=weights
        )
        self.cross_attention_layers = config.cross_attention_layers

        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                self.layers.append(
                    MllamaCrossAttentionDecoderLayer(
                        prefix=f"{prefix}.layers.{layer_idx}",
                        config=config,
                        weights=weights,
                    )
                )
            else:
                self.layers.append(
                    MllamaSelfAttentionDecoderLayer(
                        prefix=f"{prefix}.layers.{layer_idx}",
                        config=config,
                        weights=weights,
                    )
                )

        # TODO Should we use this slow norm ?
        # self.norm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        # TODO Anything specific ?
        head_size = config.hidden_size // config.num_attention_heads
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=head_size,
            base=config.rope_theta,
            device=weights.device,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.FloatTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values,
        # )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers

        for idx, decoder_layer in enumerate(self.layers):
            if (
                idx in self.cross_attention_layers
                and cross_attention_states is None
                and (
                    past_key_values is None
                    or (
                        past_key_values is not None
                        and past_key_values.get_seq_length(idx) == 0
                    )
                )
            ):
                continue

            layer_outputs = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                cross_attention_mask=cross_attention_mask,
                attention_mask=causal_mask,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states

    # def _update_causal_mask(
    #     self,
    #     attention_mask: torch.Tensor,
    #     input_tensor: torch.Tensor,
    #     cache_position: torch.Tensor,
    #     past_key_values,
    # ):
    #     if self.config._attn_implementation == "flash_attention_2":
    #         if attention_mask is not None and 0.0 in attention_mask:
    #             return attention_mask
    #         return None

    #     # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    #     # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    #     # to infer the attention mask.
    #     past_seen_tokens = (
    #         past_key_values.get_seq_length() if past_key_values is not None else 0
    #     )
    #     using_static_cache = isinstance(past_key_values, StaticCache)

    #     # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    #     # TODO: we have only SDPA currently and there's a bug when attn-bias is passed. Need to add eager attn and return the line
    #     # self.config._attn_implementation == "sdpa" and
    #     if (
    #         self.config._attn_implementation == "sdpa"
    #         and not using_static_cache
    #         and not output_attentions
    #     ):
    #         if AttentionMaskConverter._ignore_causal_mask_sdpa(
    #             attention_mask,
    #             inputs_embeds=input_tensor,
    #             past_key_values_length=past_seen_tokens,
    #             is_training=self.training,
    #         ):
    #             return None

    #     dtype, device = input_tensor.dtype, input_tensor.device
    #     min_dtype = torch.finfo(dtype).min
    #     sequence_length = input_tensor.shape[1]
    #     if using_static_cache:
    #         target_length = past_key_values.get_max_length()
    #     else:
    #         target_length = (
    #             attention_mask.shape[-1]
    #             if isinstance(attention_mask, torch.Tensor)
    #             else past_seen_tokens + sequence_length + 1
    #         )

    #     # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    #     causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
    #         attention_mask,
    #         sequence_length=sequence_length,
    #         target_length=target_length,
    #         dtype=dtype,
    #         device=device,
    #         min_dtype=min_dtype,
    #         cache_position=cache_position,
    #         batch_size=input_tensor.shape[0],
    #     )

    #     if (
    #         self.config._attn_implementation == "sdpa"
    #         and attention_mask is not None
    #         and attention_mask.device.type == "cuda"
    #         and not output_attentions
    #     ):
    #         # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    #         # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    #         # Details: https://github.com/pytorch/pytorch/issues/110213
    #         causal_mask = AttentionMaskConverter._unmask_unattended(
    #             causal_mask, min_dtype
    #         )

    #     return causal_mask


class MllamaForCausalLM(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.model = MllamaTextModel(
            prefix=f"{prefix}.model", config=config, weights=weights
        )
        self.lm_head = SpeculativeHead.load(
            prefix=f"{prefix}.lm_head",
            config=config,
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = outputs
        # if lm_head_indices is not None:
        #     hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {
                "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
                "inputs_embeds": None,
            }

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class MllamaForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        config.vision_config.quantize = None
        config.vision_config.speculator = config.speculator
        config.text_config.quantize = config.quantize
        config.text_config.speculator = config.speculator
        self.vision_model = MllamaVisionModel(
            prefix="vision_model", config=config.vision_config, weights=weights
        )
        self.language_model = MllamaForCausalLM(
            prefix="language_model", config=config.text_config, weights=weights
        )
        self.multi_modal_projector = FastLinear.load(
            prefix="multi_modal_projector", config=config, weights=weights, bias=True
        )
        self.config = config
