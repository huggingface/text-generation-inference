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

from typing import List, Optional, Tuple, Type

import torch
import torch.distributed
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from text_generation_server.layers import (
    FastLinear,
    SpeculativeHead,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    get_linear,
)
from text_generation_server.layers.attention import (
    Seqlen,
    attention,
    paged_attention,
    set_block_mapping,
    HPUPagedAttentionMetadata,
)
from text_generation_server.layers.attention.kv_cache import get_kv_scales
from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.layers.moe import DenseMoELayer, MoELayer, SparseMoELayer
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.utils.weights import UnquantizedWeight
import habana_frameworks.torch as htorch


class MixtralConfig(PretrainedConfig):
    model_type = "mixtral"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=None,
        num_experts_per_tok=2,
        num_local_experts=8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


def load_attention(config, prefix: str, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
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

    return TensorParallelColumnLinear(get_linear(weight, bias=None))


def _load_experts(config, prefix: str, mat, weights):
    if config.quantize is not None:
        raise NotImplementedError("Mixtral does not support weight quantization yet.")

    assert mat in ["w1", "w2", "w3"]

    world_size = weights.process_group.size()
    rank = weights.process_group.rank()

    assert (
        config.intermediate_size % world_size == 0
    ), f"The chosen size {config.intermediate_size} is not compatible with sharding on {world_size} shards"

    block_size = config.intermediate_size // world_size
    start = rank * block_size
    stop = (rank + 1) * block_size

    tensor = torch.empty(
        (config.num_local_experts * block_size, config.hidden_size),
        dtype=weights.dtype,
        device=weights.device,
    )

    for i in range(config.num_local_experts):
        slice_ = weights._get_slice(f"{prefix}.{i}.{mat}.weight")

        if mat == "w2":
            expert_slice = slice_[:, start:stop].t().contiguous()
        else:
            expert_slice = slice_[start:stop]
        tensor[i * block_size : (i + 1) * block_size] = expert_slice.to(
            dtype=weights.dtype
        ).to(device=weights.device)
    return tensor


class MixtralAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
        rotary_emb,
    ):
        super().__init__()
        self.max_past = (
            config.sliding_window if config.sliding_window is not None else -1
        )
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

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
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
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        kv_cache.store(
            key=kv[:, 0],
            value=kv[:, 1],
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            attn_output = attention(
                query=query,
                key=kv[:, 0],
                value=kv[:, 1],
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                softmax_scale=self.softmax_scale,
                window_size_left=self.max_past,
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

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


@torch.jit.script
def select_experts(gate_logits: torch.Tensor, top_k: int):
    # all_probs: (sequence_length, n_experts) and upcast for softmax
    all_probs = torch.nn.functional.softmax(gate_logits, dim=1, dtype=torch.float)
    # weights, selected_experts: (sequence_length, top-k)
    weights, selected_experts = torch.topk(all_probs, top_k, dim=-1)
    weights /= weights.sum(dim=-1, keepdim=True)
    weights = weights.view(-1)
    selected_experts = selected_experts.view(-1)

    return selected_experts, weights


@torch.jit.script
def round_up(x: torch.Tensor, value: int):
    return torch.div(x + (value - 1), value, rounding_mode="trunc") * value


class MixtralMoE(nn.Module):
    def __init__(
        self, prefix, config: MixtralConfig, moe_layer_cls: Type[MoELayer], weights
    ):
        super().__init__()

        # gating
        self.gate = FastLinear.load(config, f"{prefix}.gate", weights, bias=False)

        self.moe = moe_layer_cls(
            n_expert_group=None,
            n_experts=config.num_local_experts,
            prefix=f"{prefix}.experts",
            renormalize=True,
            topk=config.num_experts_per_tok,
            topk_group=None,
            weights=weights,
            gate_proj_name="w1",
            up_proj_name="w3",
            down_proj_name="w2",
        )
        assert isinstance(self.moe, MoELayer)

        self.process_group = weights.process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(x)
        out = self.moe(x, gating_output=router_logits)

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out.view(*x.shape)


class MixtralLayer(nn.Module):
    def __init__(self, prefix: str, layer_id, config, weights, rotary_emb):
        super().__init__()
        prefix = f"{prefix}.layers.{layer_id}"

        self.self_attn = MixtralAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            rotary_emb=rotary_emb,
        )

        moe_layer_cls = (
            SparseMoELayer if SparseMoELayer.is_supported(weights) else DenseMoELayer
        )
        self.moe = MixtralMoE(
            f"{prefix}.block_sparse_moe", config, moe_layer_cls, weights
        )

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

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        moe_output = self.moe(normed_attn_res_output)

        return moe_output, attn_res


class MixtralModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        self.embed_tokens = TensorParallelEmbedding(
            prefix=(
                "model.embed_tokens" if not prefix else f"{prefix}.model.embed_tokens"
            ),
            weights=weights,
        )

        rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
            device=weights.device,
        )
        self.layers = nn.ModuleList(
            [
                MixtralLayer(
                    "model" if not prefix else f"{prefix}.model",
                    layer_id,
                    config,
                    weights,
                    rotary_emb,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix="model.norm" if not prefix else f"{prefix}.model.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

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
        seqlen: Seqlen,
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


class FlashMixtralForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        self.model = MixtralModel(prefix, config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="lm_head" if not prefix else f"{prefix}.lm_head",
            weights=weights,
        )
        self.max_past = config.sliding_window
        self.max_past_tensor = (
            torch.tensor(config.sliding_window, device=weights.device)
            if self.max_past is not None
            else None
        )

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
    ) -> torch.Tensor:
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
        logits = self.lm_head(hidden_states)
        return logits
