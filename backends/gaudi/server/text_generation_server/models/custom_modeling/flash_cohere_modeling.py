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

from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    set_block_mapping,
    Seqlen,
    HPUPagedAttentionMetadata,
)
from text_generation_server.layers.attention.kv_cache import get_kv_scales
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
from text_generation_server.utils.weights import UnquantizedWeight
from habana_frameworks.torch.hpex.kernels import (
    RotaryPosEmbeddingMode,
    apply_rotary_pos_emb,
)

import habana_frameworks.torch as htorch


class CohereRotary(PositionRotaryEmbedding):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # Such controlflows may add some overhead.
        num_tokens = query.shape[0]
        head_size = query.shape[-1]
        rope_mode = RotaryPosEmbeddingMode.PAIRWISE
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        rotary_dim = cos.shape[-1]
        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query.copy_(torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape))

        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key.copy_(torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape))


class CohereLayerNorm(nn.Module):
    def __init__(self, prefix, weights, eps):
        super().__init__()
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)
        self.weight = nn.Parameter(weight)
        # Fake weights
        self.ones = weight.new_ones(weight.shape[1])
        self.eps = eps

    def forward(self, hidden_states):
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
        dim=0,
    )

    if isinstance(weight, UnquantizedWeight):
        weight.weight = weight.weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    if config.attention_bias:
        w = [
            weights.get_sharded(f"{p}.bias", dim=0)
            for p in [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        ]
        bias = torch.cat(w, dim=0).to(dtype=weights.dtype).to(device=weights.device)
    else:
        bias = None

    return TensorParallelColumnLinear(get_linear(weight, bias=bias))


class FlashCohereAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
        rotary_emb,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = rotary_emb

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
        self.kv_scales = get_kv_scales(weights, f"{prefix}")

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
        slots,
        seqlen,
        hpu_attention_meta,
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

        kv_cache.store(
            key=key,
            value=value,
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            attn_output = attention(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                softmax_scale=self.softmax_scale,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                seqlen,
                kv_scales=self.kv_scales,
                hpu_attention_meta=hpu_attention_meta,
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
    def __init__(self, prefix: str, layer_id, config, weights, rotary_emb):
        super().__init__()
        prefix = f"{prefix}.layers.{layer_id}"
        self.self_attn = FlashCohereAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            rotary_emb=rotary_emb,
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
        slots,
        seqlen,
        hpu_attention_meta,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )

        mlp_output = self.mlp(normed_hidden_states)
        output = attn_output + mlp_output

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(output, group=self.process_group)

        return output, res


class FlashCohereModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens", weights=weights
        )
        rotary_emb = CohereRotary.static(
            config=config,
            dim=config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
            device=weights.device,
        )
        self.layers = nn.ModuleList(
            [
                FlashCohereLayer(
                    prefix,
                    layer_id,
                    config,
                    weights,
                    rotary_emb,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastLayerNorm.load_no_bias(
            prefix=f"{prefix}.norm", weights=weights, eps=config.layer_norm_eps
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
        slots: torch.Tensor,
        seqlen: torch.Tensor,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
    ) -> torch.Tensor:
        if hpu_attention_meta is not None:
            hpu_attention_meta = set_block_mapping(
                hpu_attention_meta, input_ids.shape[0]
            )
        hidden_states = self.embed_tokens(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(position_ids)

        residual = None
        lazy_mode = htorch.utils.internal.is_lazy()
        if lazy_mode:
            htorch.core.mark_step()
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                slots,
                seqlen,
                hpu_attention_meta,
            )
            if lazy_mode:
                htorch.core.mark_step()

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashCohereForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        if not prefix:
            prefix = "model"
        else:
            prefix = f"{prefix}.model"

        self.model = FlashCohereModel(prefix, config, weights)
        try:
            self.lm_head = SpeculativeHead.load(
                config,
                prefix="lm_head",
                weights=weights,
            )
        except RuntimeError:
            self.lm_head = SpeculativeHead.load(
                config,
                prefix=f"{prefix}.embed_tokens",
                weights=weights,
            )
        self.logit_scale = config.logit_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        logits *= self.logit_scale
        if speculative_logits is not None:
            speculative_logits *= self.logit_scale
        return logits, speculative_logits
