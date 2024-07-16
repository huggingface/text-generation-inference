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
""" PyTorch Idefics2 model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import math

from transformers.activations import ACT2FN
from transformers.image_processing_utils import select_best_resolution
from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
    load_vision_model,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from text_generation_server.utils.weights import DefaultWeightsLoader


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Idefics2VisionEmbeddings(nn.Module):
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings` to enable images of variable
    resolution.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
    which allows treating images in their native aspect ratio and without the need to resize them to the same
    fixed size. In particular, we start from the original pre-trained SigLIP model
    (which uses images of fixed-size square images) and adapt it by training on images of variable resolutions.
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.patch_embedding.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embedding.weight"), requires_grad=False
        )
        self.patch_embedding.bias = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embedding.bias"), requires_grad=False
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = TensorParallelEmbedding(
            prefix=f"{prefix}.position_embedding", weights=weights
        )

    def forward(
        self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
    ) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(
                fractional_coords_h, boundaries, right=True
            )
            bucket_coords_w = torch.bucketize(
                fractional_coords_w, boundaries, right=True
            )

            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Idefics2VisionAttention(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.embed_dim // self.num_heads
        if self.head_size * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_size**-0.5
        self.dropout = config.attention_dropout

        self.num_heads = self.num_heads // weights.process_group.size()
        self.embed_dim = self.embed_dim // weights.process_group.size()

        self.qkv = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )
        self.out_proj = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.out_proj", weights=weights, bias=True
        )
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv.split(
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
            ],
            dim=2,
        )

        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_size):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_size)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class Idefics2VisionMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.fc1", config=config, weights=weights, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(
            prefix=f"{prefix}.fc2", config=config, weights=weights, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Idefics2EncoderLayer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Idefics2VisionAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.layer_norm1 = nn.LayerNorm.load(
            prefix=f"{prefix}.layer_norm1", eps=config.layer_norm_eps, weights=weights
        )
        self.layer_norm2 = nn.LayerNorm.load(
            prefix=f"{prefix}.layer_norm2", eps=config.layer_norm_eps, weights=weights
        )
        self.mlp = Idefics2VisionMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights
        )

    # Copied from transformers.models.siglip.modeling_siglip.SiglipEncoderLayer.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Idefics2Encoder(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Idefics2EncoderLayer(
                    prefix=f"{prefix}.layers.{i}", config=config, weights=weights
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
            )
        return hidden_states


class Idefics2VisionTransformer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embeddings = Idefics2VisionEmbeddings(
            prefix=f"{prefix}.embeddings", config=config, weights=weights
        )
        self.encoder = Idefics2Encoder(
            prefix=f"{prefix}.encoder", config=config, weights=weights
        )
        self.post_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.post_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
    ):
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(
                dtype=torch.bool, device=pixel_values.device
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        else:
            patch_attention_mask = _prepare_4d_attention_mask(
                patch_attention_mask, hidden_states.dtype
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
        )

        last_hidden_state = encoder_outputs
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class Idefics2MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.text_config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
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

    def forward(self, hidden_states):
        start_shape = hidden_states.shape[:-1]
        gate_up_states = self.gate_up_proj(hidden_states)
        intermediate_size = gate_up_states.shape[-1] // 2
        gate_up_states = gate_up_states.view(-1, 2, intermediate_size)
        return self.down_proj(
            self.act(gate_up_states[:, 0]) * gate_up_states[:, 1]
        ).view(*start_shape, -1)


class Idefics2RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps):
        """
        Idefics2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.weight"), requires_grad=False
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.layer_idx = None
        self.hidden_size = config.text_config.hidden_size
        self.num_heads = config.perceiver_config.resampler_n_heads
        self.head_size = config.perceiver_config.resampler_head_dim
        self.num_key_value_heads = config.perceiver_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.perceiver_config.attention_dropout
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            self.num_key_value_heads // weights.process_group.size()
        )

        self.q_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.q_proj",
            weights=weights,
            bias=False,
        )
        self.kv = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.o_proj", weights=weights, bias=False
        )

        self.is_causal = False

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        hidden_states = torch.concat([context, latents], dim=-2)
        query_states = self.q_proj(latents)
        kv = self.kv(hidden_states)
        key_states, value_states = kv.split(
            [
                self.head_size * self.num_key_value_heads,
                self.head_size * self.num_key_value_heads,
            ],
            dim=2,
        )

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_size
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_size
        ).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_size)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_size)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        self.input_latents_norm = Idefics2RMSNorm(
            prefix=f"{prefix}.input_latents_norm",
            weights=weights,
            eps=self.rms_norm_eps,
        )
        self.input_context_norm = Idefics2RMSNorm(
            prefix=f"{prefix}.input_context_norm",
            weights=weights,
            eps=self.rms_norm_eps,
        )
        self.self_attn = Idefics2PerceiverAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.post_attention_layernorm = Idefics2RMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=self.rms_norm_eps,
        )
        self.mlp = Idefics2MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
        """
        residual = latents

        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        latents = self.self_attn(
            latents=latents,
            context=context,
            attention_mask=attention_mask,
        )
        latents = residual + latents
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        return latents


class Idefics2PerceiverResampler(nn.Module):
    def __init__(self, prefix, config, weights) -> None:
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.hidden_act = config.perceiver_config.hidden_act
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        # Create Latents for Perceiver
        self.latents = weights.get_tensor(f"{prefix}.latents")

        # Create Transformer Blocks
        self.layers = nn.ModuleList(
            [
                Idefics2PerceiverLayer(
                    prefix=f"{prefix}.layers.{idx}", config=config, weights=weights
                )
                for idx in range(self.depth)
            ]
        )
        self.norm = Idefics2RMSNorm(
            prefix=f"{prefix}.norm",
            weights=weights,
            eps=config.text_config.rms_norm_eps,
        )

    def forward(
        self,
        context: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        # seq embed -> bsz seq embed
        latents = self.latents.unsqueeze(0).expand(
            (context.shape[0], *self.latents.size())
        )

        latent_attention_mask = torch.ones(
            (attention_mask.size(0), latents.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, latent_attention_mask], dim=-1)
        attention_mask = _prepare_4d_attention_mask(
            attention_mask, latents.dtype, tgt_len=self.n_latents
        )

        compressed_context = latents
        for perceiver_layer in self.layers:
            compressed_context = perceiver_layer(
                compressed_context,
                context,
                attention_mask=attention_mask,
            )
        compressed_context = self.norm(compressed_context)

        return compressed_context


class Idefics2Connector(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.modality_projection = Idefics2MLP(
            prefix=f"{prefix}.modality_projection", config=config, weights=weights
        )
        self.perceiver_resampler = Idefics2PerceiverResampler(
            prefix=f"{prefix}.perceiver_resampler", config=config, weights=weights
        )

    def forward(self, image_hidden_states, attention_mask):
        image_hidden_states = self.modality_projection(image_hidden_states)
        image_hidden_states = self.perceiver_resampler(
            context=image_hidden_states, attention_mask=attention_mask
        )
        return image_hidden_states


class Idefics2ForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        config.vision_config.quantize = None
        config.vision_config.speculator = config.speculator
        config.text_config.quantize = config.quantize
        config.text_config.speculator = config.speculator

        vision_config = config.vision_config
        self.text_model = load_text_model(
            prefix="model" if not prefix else f"{prefix}.model",
            config=config.text_config,
            weights=weights,
            name="text_model",
        )
        self.dtype = weights.dtype

        # The vision and connector models are not quantized.
        with weights.use_loader(DefaultWeightsLoader()):
            self.vision_model = Idefics2VisionTransformer(
                prefix=(
                    f"{prefix}.model.vision_model" if prefix else "model.vision_model"
                ),
                config=vision_config,
                weights=weights,
            )

            quantize = config.quantize
            try:
                config.quantize = None
                self.connector = Idefics2Connector(
                    prefix=f"{prefix}.model.connector" if prefix else "model.connector",
                    config=config,
                    weights=weights,
                )
            finally:
                config.quantize = quantize

        self.config = config
        self.image_seq_len = config.perceiver_config.resampler_n_latents
        self.image_token_id = config.image_token_id
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ):
        """In place merges in vision_embeddings with inputs_embeds."""
        # mask = input_ids == self.config.image_token_index
        mask = input_ids == self.config.image_token_id
        # Let's pray we have enabled enough slots !
        inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor] = None,
        pixel_values: torch.FloatTensor = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        # Unused here
        image_sizes: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        if pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            all_states = []
            all_pixel_values = pixel_values
            all_pixel_mask = pixel_attention_mask
            for i in range(batch_size):
                pixel_values = all_pixel_values.to(
                    dtype=self.dtype
                )  # fp16 compatibility
                pixel_values = pixel_values[i : i + 1]
                pixel_values = pixel_values.view(num_images, *pixel_values.shape[2:])

                # Remove padding images - padding images are full 0.
                nb_values_per_image = pixel_values.shape[1:].numel()
                real_images_inds = (pixel_values == 0.0).sum(
                    dim=(-1, -2, -3)
                ) != nb_values_per_image
                pixel_values = pixel_values[real_images_inds].contiguous()

                # Handle the vision attention mask
                if pixel_attention_mask is None:
                    pixel_attention_mask = torch.ones(
                        size=(
                            pixel_values.size(0),
                            pixel_values.size(2),
                            pixel_values.size(3),
                        ),
                        dtype=torch.bool,
                        device=pixel_values.device,
                    )
                else:
                    # Remove padding images from the mask/pP p
                    pixel_attention_mask = all_pixel_mask[i : i + 1]
                    pixel_attention_mask = pixel_attention_mask.view(
                        1 * num_images, *pixel_attention_mask.shape[2:]
                    )
                    pixel_attention_mask = pixel_attention_mask[
                        real_images_inds
                    ].contiguous()

                patch_size = self.config.vision_config.patch_size
                patches_subgrid = pixel_attention_mask.unfold(
                    dimension=1, size=patch_size, step=patch_size
                )
                patches_subgrid = patches_subgrid.unfold(
                    dimension=2, size=patch_size, step=patch_size
                )
                patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

                # Get sequence from the vision encoder
                image_hidden_states = self.vision_model(
                    pixel_values=pixel_values,
                    patch_attention_mask=patch_attention_mask,
                )

                # Modality projection & resampling
                image_hidden_states = self.connector(
                    image_hidden_states,
                    attention_mask=patch_attention_mask.view(pixel_values.size(0), -1),
                )
                all_states.append(image_hidden_states)
            image_hidden_states = torch.stack(all_states, dim=0)
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self._merge_input_ids_with_image_features(
                input_ids, inputs_embeds, image_hidden_states
            )

        hidden_states = self.text_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            true_max_s=max_s,
            prefill_cache_indices=None,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.text_model.lm_head(hidden_states)
        return logits, speculative_logits
