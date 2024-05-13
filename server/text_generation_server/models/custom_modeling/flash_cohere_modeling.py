# coding=utf-8
# Copyright 2024 Cohere team. All rights reserved.
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
from typing import Optional, List, Tuple

from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.layernorm import (
    FastLayerNorm,
)
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
)

if SYSTEM == "cuda":
    import dropout_layer_norm
else:
    dropout_layer_norm = None


class CohereRotary(PositionRotaryEmbedding):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # Such controlflows may add some overhead.
        if SYSTEM == "cuda":
            import rotary_emb

            q1 = query[..., ::2]
            q2 = query[..., 1::2]

            rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)

            k1 = key[..., ::2]
            k2 = key[..., 1::2]

            rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
        elif SYSTEM == "rocm":
            from vllm import pos_encoding_ops

            # NOTE: On RoCm systems, we use a ROPE implementatation adapted from VLLM which launches a single kernel for both query/key, contrary to flash-attn implementation used on NVIDIA systems.
            # Compiling flash-attn rotary on RoCm, it appears hipcc is unable to unroll loops, resulting in an even slower inference compared to eager: https://github.com/pytorch/pytorch/issues/113773

            head_size = query.shape[-1]

            # Inplace operation, updating query and key.
            pos_encoding_ops.rotary_embedding(query, key, head_size, cos, sin, False)
        else:
            raise ValueError(
                "Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction."
            )


class CohereLayerNorm(nn.Module):
    def __init__(self, prefix, weights, eps):
        super().__init__()
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)
        self.weight = nn.Parameter(weight)
        # Fake weights
        self.ones = weight.new_ones(weight.shape[1])
        self.eps = eps

    def forward(self, hidden_states):
        if hidden_states.shape[-1] > 8192 or SYSTEM == "rocm":
            hidden_states = hidden_states.reshape(
                -1, self.weight.shape[0], self.weight.shape[1]
            )
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            mean = hidden_states.mean(-1, keepdim=True)
            hidden_states_minus_mean = hidden_states - mean
            variance = hidden_states_minus_mean.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states_minus_mean * torch.rsqrt(variance + self.eps)
            hidden_states = self.weight.to(torch.float32) * hidden_states
            hidden_states = hidden_states.view(-1, self.weight.shape[1])
            return hidden_states.to(input_dtype)

        (
            hidden_states,
            *rest,
        ) = dropout_layer_norm.dropout_add_ln_fwd(
            hidden_states,
            None,
            self.ones,
            None,
            None,
            None,
            None,
            None,
            0.0,
            self.eps,
            1.0,
            0,
            None,
            False,
            False,
        )

        # Required to apply one weight matrix per head
        hidden_states = hidden_states.view(
            -1, self.weight.shape[0], self.weight.shape[1]
        )
        hidden_states = self.weight * hidden_states
        hidden_states = hidden_states.view(-1, self.weight.shape[1])

        return hidden_states


def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=config.attention_bias,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    if config.attention_bias:
        w = [
            weights.get_sharded(f"{p}.bias", dim=0)
            for p in [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        ]
        bias = torch.cat(w, dim=0).to(dtype=weights.dtype).to(device=weights.device)
    else:
        bias = None

    return TensorParallelColumnLinear(
        get_linear(weight, bias=bias, quantize=config.quantize)
    )


class FlashCohereAttention(torch.nn.Module):
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

        self.rotary_emb = CohereRotary.static(
            config=config,
            dim=self.head_size,
            base=config.rope_theta,
            device=weights.device,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )

        self.query_key_value = load_attention(config, prefix, weights)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = CohereLayerNorm(
                prefix=f"{prefix}.q_norm",
                weights=weights,
                eps=config.layer_norm_eps,
            )
            self.k_norm = CohereLayerNorm(
                prefix=f"{prefix}.k_norm",
                weights=weights,
                eps=config.layer_norm_eps,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=config.attention_bias,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        qkv = self.query_key_value(hidden_states)
        query, key, value = qkv.split(
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_key_value_heads,
                self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )

        if self.use_qk_norm:
            query = query.reshape(-1, self.head_size)
            key = key.reshape(-1, self.head_size)
            query = self.q_norm(query.contiguous())
            key = self.k_norm(key.contiguous())

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_key_value_heads, self.head_size)
        value = value.view(-1, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, key, cos, sin)

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

        return self.o_proj(
            attn_output.view(-1, self.num_heads * self.head_size), reduce=False
        )


class CohereMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
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
        # Fuse gate and up proj
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
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(
            self.act(gate_up_states[:, 0]) * gate_up_states[:, 1], reduce=False
        )


class FlashCohereLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashCohereAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = CohereMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = FastLayerNorm.load_no_bias(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )
        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        mlp_output = self.mlp(normed_hidden_states)
        output = attn_output + mlp_output

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(output, group=self.process_group)

        return output, res


class FlashCohereModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashCohereLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastLayerNorm.load_no_bias(
            prefix="model.norm", weights=weights, eps=config.layer_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

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
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashCohereForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashCohereModel(config, weights)
        try:
            self.lm_head = SpeculativeHead.load(
                config,
                prefix="lm_head",
                weights=weights,
            )
        except RuntimeError:
            self.lm_head = SpeculativeHead.load(
                config,
                prefix="model.embed_tokens",
                weights=weights,
            )
        self.logit_scale = config.logit_scale

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
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        logits *= self.logit_scale
        if speculative_logits is not None:
            speculative_logits *= self.logit_scale
        return logits, speculative_logits
