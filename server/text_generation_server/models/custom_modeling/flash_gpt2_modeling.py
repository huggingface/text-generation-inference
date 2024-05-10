# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
    FastLayerNorm,
)


def _load_qkv(config, prefix: str, weights, head_size, num_heads):
    slice_ = weights._get_slice(f"{prefix}.c_attn.weight")
    shape = slice_.get_shape()
    total_size = shape[1]
    assert total_size % 3 == 0, f"Prepacked is not divisible by {3}"
    world_size = weights.process_group.size()
    single_size = total_size // 3
    rank = weights.process_group.rank()

    # Weights
    block_size = single_size // world_size
    start = rank * block_size
    stop = (rank + 1) * block_size
    tensors = []
    for i in range(3):
        tensor = slice_[:, start + i * single_size : stop + i * single_size]
        tensors.append(tensor)
    weight = torch.cat(tensors, dim=1).T
    weight = weight.to(device=weights.device)
    weight = weight.to(dtype=weights.dtype)

    # Bias
    slice_ = weights._get_slice(f"{prefix}.c_attn.bias")
    shape = slice_.get_shape()
    total_size = shape[0]
    single_size = total_size // 3
    block_size = single_size // world_size
    assert single_size % world_size == 0
    start = rank * block_size
    stop = (rank + 1) * block_size
    b = []
    for i in range(3):
        tensor = slice_[start + i * single_size : stop + i * single_size]
        b.append(tensor)
    bias = torch.cat(b, dim=0)
    bias = bias.to(dtype=weights.dtype).to(device=weights.device)
    assert list(bias.shape) == [
        3 * num_heads * head_size
    ], f"{weight.shape} != {[3 * num_heads * head_size]}"

    return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))


def load_row(config, transpose: bool, prefix: str, weights, bias: bool):
    if transpose:
        weight = weights.get_sharded(f"{prefix}.weight", dim=0).T
    else:
        weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None
    return TensorParallelRowLinear(
        get_linear(weight, bias, config.quantize), process_group=weights.process_group
    )


class FlashGPT2Attention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = self.num_heads // weights.process_group.size()

        self.query_key_value = _load_qkv(
            config,
            prefix=prefix,
            weights=weights,
            head_size=self.head_size,
            num_heads=self.num_heads,
        )

        self.o_proj = load_row(
            config,
            transpose=True,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=True,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

    def forward(
        self,
        hidden_states,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        query, key, value = self.query_key_value(hidden_states).split(
            self.head_size * self.num_heads, dim=1
        )
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)

        paged_attention.reshape_and_cache(key, value, kv_cache[0], kv_cache[1], slots)

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
                query,
                key,
                value,
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            paged_attention.attention(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class GPT2MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.activation_function
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

        self.c_fc = load_row(
            config, prefix=f"{prefix}.c_fc", weights=weights, transpose=True, bias=True
        )
        self.c_proj = load_row(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            transpose=True,
            bias=True,
        )

        intermediate_size = (
            config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        )

        self.intermediate_size = intermediate_size // weights.process_group.size()

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.c_proj(hidden_states)


class FlashGPT2Layer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.self_attn = FlashGPT2Attention(
            prefix=f"{prefix}.attn", config=config, weights=weights
        )
        self.mlp = GPT2MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
        )
        self.post_attention_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.ln_2",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        hidden_states = attn_output + residual
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(hidden_states)

        return residual + mlp_output, residual


class FlashGPT2Model(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.layers = nn.ModuleList(
            [
                FlashGPT2Layer(
                    prefix=(
                        f"h.{layer_id}" if not prefix else f"{prefix}.h.{layer_id}"
                    ),
                    config=config,
                    weights=weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.norm = nn.LayerNorm.load(
            prefix="ln_f" if not prefix else f"{prefix}.ln_f",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        true_max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = inputs_embeds

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class FlashGPT2ForCausalLM(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.embed_tokens = TensorParallelEmbedding(
            prefix=("wte" if not prefix else f"{prefix}.wte"),
            weights=weights,
        )
        self.embed_positions = TensorParallelEmbedding(
            prefix=("wpe" if not prefix else f"{prefix}.wpe"),
            weights=weights,
        )

        self.model = FlashGPT2Model(prefix, config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="wte" if not prefix else f"{prefix}.wte",
            weights=weights,
        )

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
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        token_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        inputs_embeds = token_embeds + position_embeds
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            true_max_s=max_s,
            prefill_cache_indices=prefill_cache_indices,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
