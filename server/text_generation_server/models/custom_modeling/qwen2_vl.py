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
"""PyTorch Qwen2 VL model."""

from typing import Optional, Tuple, List

import torch
import torch.utils.checkpoint
from torch import nn
from text_generation_server.utils.import_utils import SYSTEM

if SYSTEM == "ipex":
    pass
else:
    pass

from transformers.activations import ACT2FN
import torch.nn.functional as F

from text_generation_server.layers.layernorm import FastLayerNorm, FastRMSNorm
from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorParallelEmbedding,
    FastLinear,
)
from text_generation_server.layers.attention import (
    Seqlen,
)
from text_generation_server.models.custom_modeling.flash_qwen2_modeling import (
    Qwen2Model,
)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class Qwen2VLSdpaAttention(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.head_dim = config.hidden_size // config.num_heads
        self.num_heads = config.num_heads // weights.process_group.size()

        self.qkv = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.qkv",
            weights=weights,
            bias=False,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
        )
        self.qkv.linear.bias = weights.get_sharded(f"{prefix}.qkv.bias", dim=0)
        self.proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.proj",
            weights=weights,
            bias=True,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # apply the qkv linear layer to the hidden state
        qkv = self.qkv(hidden_state)
        query, key, value = qkv.split(
            [self.embed_dim, self.embed_dim, self.embed_dim], dim=1
        )

        # reshape the query, key, and value tensors
        _shape = (
            hidden_state.shape[0],
            self.num_heads,
            self.embed_dim // self.num_heads,
        )
        query = query.view(*_shape)
        key = key.view(*_shape)
        value = value.view(*_shape)

        # apply rotary positional embeddings
        query = apply_rotary_pos_emb_vision(query.unsqueeze(0), rotary_pos_emb).squeeze(
            0
        )
        key = apply_rotary_pos_emb_vision(key.unsqueeze(0), rotary_pos_emb).squeeze(0)
        # TODO: make use of existing RotatoryPositionEmbedding class

        # create the attention mask
        attention_mask = torch.zeros(
            [1, hidden_state.shape[0], hidden_state.shape[0]],
            device=hidden_state.device,
            dtype=torch.bool,
        )
        # TODO: avoid creating the mask in the forward pass, instead define the largest possible mask and slice it

        # apply the cu_seqlens to the attention mask
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = True

        # transpose for the attention mechanism (batch, seqlen, hidden_dim) -> (seqlen, batch, hidden_dim)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        # apply attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(hidden_state.shape[0], -1)
        # TODO: prefer flash attention

        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VLVisionMLP(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
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


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.attn = Qwen2VLSdpaAttention(
            prefix=f"{prefix}.attn",
            config=config,
            weights=weights,
        )
        self.norm1 = FastLayerNorm.load(
            prefix=f"{prefix}.norm1",
            weights=weights,
            eps=1e-6,
        )
        self.norm2 = FastLayerNorm.load(
            prefix=f"{prefix}.norm2",
            weights=weights,
            eps=1e-6,
        )
        self.mlp = Qwen2VLVisionMLP(
            prefix=f"{prefix}.mlp",
            config=config,
            weights=weights,
        )

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states_post_norm1, res = self.norm1(hidden_states)
        hidden_states = hidden_states + self.attn(
            hidden_states_post_norm1, cu_seqlens, rotary_pos_emb
        )
        hidden_states_post_norm2, res = self.norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(hidden_states_post_norm2)
        return hidden_states


class Qwen2VLPatchMerger(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.embed_dim * (config.spatial_merge_size**2)
        self.patch_merger_ln_q = FastLayerNorm.load(
            prefix=f"{prefix}.ln_q",
            weights=weights,
            eps=1e-6,
        )
        self.fc1 = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.mlp.0", weights=weights, config=config, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(
            prefix=f"{prefix}.mlp.2", weights=weights, config=config, bias=True
        )

    def forward(self, hidden_states, grid_thw) -> torch.Tensor:
        hidden_states, _ = self.patch_merger_ln_q(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Qwen2VisionModel(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        kernel_size = [config.temporal_patch_size, config.patch_size, config.patch_size]
        self.patch_embedding = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=config.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )
        self.patch_embedding.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embed.proj.weight"), requires_grad=False
        )
        head_dim = config.embed_dim // config.num_heads
        # TODO: replace with static positional embeddings once implemented
        theta = 10000.0
        dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.blocks = nn.ModuleList(
            [
                Qwen2VLVisionBlock(
                    prefix=f"{prefix}.blocks.{i}",
                    config=config,
                    weights=weights,
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Qwen2VLPatchMerger(
            prefix=f"{prefix}.merger",
            config=config,
            weights=weights,
        )

        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_patch_size = config.spatial_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # reshape the input tensor for processing
        shape = (
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.spatial_patch_size,
            self.spatial_patch_size,
        )
        pixel_values = pixel_values.view(shape).to(self.patch_embedding.weight.dtype)
        hidden_states = self.patch_embedding(pixel_values).view(-1, self.embed_dim)
        # TODO: revisit to see if we can avoid some of these reshapes

        # find the position ids for the input tensor based on the grid_thw
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()

        # apply the positional embeddings to the position ids
        seq = torch.arange(
            max_grid_size, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        rotary_pos_emb_full = torch.outer(seq, self.inv_freq)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device, hidden_states.dtype)

        # create a cu_seqlens tensor to be used in the attention mask
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # iterately apply the blocks to the hidden states
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)

        # apply the final patch merger to the hidden states
        hidden_states = self.merger(hidden_states, grid_thw)
        return hidden_states


class Qwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        config.vision_config.quantize = None
        config.vision_config.speculator = config.speculator
        self.hidden_size = config.hidden_size
        self.vision_start_token_id = config.vision_start_token_id
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.visual = Qwen2VisionModel(
            prefix="visual", config=config.vision_config, weights=weights
        )
        self.text_model = Qwen2Model(prefix=None, config=config, weights=weights)
        self.lm_head = FastLinear.load(
            prefix="lm_head", weights=weights, config=config, bias=False
        )
        self.norm = FastRMSNorm.load(
            prefix="model.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.device = weights.device

    def get_position_ids(
        self,
        batch_input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
        # video_grid_thw is not implemented yet as we do not accept video inputs at the moment
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.ones(
            3,
            batch_input_ids.shape[0],
            batch_input_ids.shape[1],
            dtype=batch_input_ids.dtype,
            device=batch_input_ids.device,
        )
        d = batch_input_ids.device
        if image_grid_thw is not None:
            image_index = 0
            llm_pos_ids_list = []

            for i, input_ids in enumerate(batch_input_ids):
                vision_start_indices = torch.argwhere(
                    input_ids == self.vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                # only copy the sum of the image tokens GPU<->CPU
                image_count = (vision_tokens == self.image_token_id).sum().item()

                current_pos = 0
                for _ in range(image_count):
                    # copy the value position of the next image token from GPU<->CPU
                    next_image_pos = (
                        (input_ids[current_pos:] == self.image_token_id)
                        .nonzero()[0]
                        .item()
                    )
                    # TODO: revisit above to get all next_image_pos in one go to avoid copying in the loop
                    time_steps, height, width = image_grid_thw[image_index].clone()
                    height //= self.spatial_merge_size
                    width //= self.spatial_merge_size

                    # calculate the length of the text and image tokens
                    text_length = next_image_pos - current_pos
                    start_idx = (
                        llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    )

                    # text position ids
                    text_pos_ids = torch.arange(text_length, device=d)
                    text_pos_ids = text_pos_ids.view(1, -1).expand(3, -1) + start_idx
                    llm_pos_ids_list.append(text_pos_ids)

                    # image position ids
                    t_indices = torch.arange(time_steps, device=d).repeat_interleave(
                        height * width
                    )
                    h_indices = (
                        torch.arange(height, device=d)
                        .repeat_interleave(width)
                        .repeat(time_steps)
                    )
                    w_indices = torch.arange(width, device=d).repeat(
                        height * time_steps
                    )

                    image_pos_ids = (
                        torch.stack([t_indices, h_indices, w_indices])
                        + text_length
                        + start_idx
                    )
                    llm_pos_ids_list.append(image_pos_ids)

                    current_pos = next_image_pos + time_steps * height * width
                    image_index += 1

            if current_pos < batch_input_ids.size(1):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = batch_input_ids.size(1) - current_pos
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=d).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[:, i, :] = llm_positions.to(position_ids.device)

        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor],
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_attention_mask=None,
        image_sizes: Optional[torch.LongTensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        image_indices=None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        # apply the visual model to the pixel values if they are provided
        if pixel_values is not None and len(pixel_values) > 0:
            if pixel_values is not None:
                image_embeds = self.visual(
                    pixel_values, grid_thw=image_grid_thw
                ).squeeze(0)
                inputs_embeds[input_ids == self.image_token_id] = image_embeds

        hidden_states = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            seqlen=seqlen,
            max_s=max_s,
            true_max_s=max_s,
            prefill_cache_indices=prefill_cache_indices,
        )
        hidden_states, _ = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, None
