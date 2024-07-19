# coding=utf-8
# Copyright 2022 HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, List, Tuple, Any
from text_generation_server.utils.import_utils import SYSTEM

if SYSTEM != "ipex":
    from vllm.model_executor.layers.fused_moe import fused_moe

from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    reshape_and_cache,
)
from text_generation_server.layers import (
    FastLinear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
)
from text_generation_server.layers.layernorm import (
    FastLayerNorm,
)
from text_generation_server.utils.log import log_once


class DbrxAttentionConfig(PretrainedConfig):
    def __init__(
        self,
        attn_pdrop: float = 0,
        clip_qkv: Optional[float] = None,
        kv_n_heads: int = 1,
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta

        for k in ["model_type"]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")


class DbrxFFNConfig(PretrainedConfig):
    def __init__(
        self,
        ffn_act_fn: Optional[dict] = None,
        ffn_hidden_size: int = 3584,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_jitter_eps: Optional[float] = None,
        moe_loss_weight: float = 0.01,
        moe_normalize_expert_weights: Optional[float] = 1,
        uniform_expert_assignment: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        if ffn_act_fn is None:
            ffn_act_fn = {"name": "silu"}
        self.ffn_act_fn = ffn_act_fn
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

        if uniform_expert_assignment:
            raise ValueError("`uniform_expert_assignment = True` is not supported")

        for k in ["model_type"]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")


class DbrxConfig(PretrainedConfig):
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        max_seq_len: int = 2048,
        vocab_size: int = 32000,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        attn_config: Optional[DbrxAttentionConfig] = None,
        ffn_config: Optional[DbrxFFNConfig] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.05,
        **kwargs: Any,
    ):
        if attn_config is None:
            self.attn_config = DbrxAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = DbrxAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config

        if ffn_config is None:
            self.ffn_config = DbrxFFNConfig()
        elif isinstance(ffn_config, dict):
            self.ffn_config = DbrxFFNConfig(**ffn_config)
        else:
            self.ffn_config = ffn_config

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        if tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported for Dbrx models.")

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_key_value_heads(self):
        # We can't use the attribute map, since this the number of KV
        # heads is not top-level.
        return self.attn_config.kv_n_heads


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


def load_attention(config, prefix, weights):
    return TensorParallelColumnLinear.load_qkv(
        config,
        prefix=f"{prefix}.Wqkv",
        weights=weights,
        bias=False,
        num_heads=config.n_heads,
        num_key_value_heads=config.attn_config.kv_n_heads,
    )


def _load_experts(config, prefix, weights):
    world_size = weights.process_group.size()
    rank = weights.process_group.rank()

    assert (
        config.ffn_config.ffn_hidden_size % world_size == 0
    ), f"The chosen size {config.ffn_config.ffn_hidden_size} is not compatible with sharding on {world_size} shards"

    expert_size = config.ffn_config.ffn_hidden_size
    block_size = expert_size // world_size
    start = rank * block_size
    stop = (rank + 1) * block_size

    tensor = torch.empty(
        (config.ffn_config.moe_num_experts * block_size, config.d_model),
        dtype=weights.dtype,
        device=weights.device,
    )

    slice_ = weights._get_slice(f"{prefix}")

    for i in range(config.ffn_config.moe_num_experts):
        offset = i * expert_size
        expert_slice = slice_[start + offset : stop + offset]

        tensor[i * block_size : (i + 1) * block_size] = expert_slice.to(
            dtype=weights.dtype
        ).to(device=weights.device)
    return tensor


def _load_experts_quantized(config, prefix, weights, cls):
    world_size = weights.process_group.size()
    rank = weights.process_group.rank()

    assert (
        config.ffn_config.ffn_hidden_size % world_size == 0
    ), f"The chosen size {config.ffn_config.ffn_hidden_size} is not compatible with sharding on {world_size} shards"

    expert_size = config.ffn_config.ffn_hidden_size
    block_size = expert_size // world_size
    start = rank * block_size
    stop = (rank + 1) * block_size

    slice_ = weights._get_slice(f"{prefix}")

    experts = []
    for i in range(config.ffn_config.moe_num_experts):
        if config.quantize in ["gptq", "awq"]:
            raise NotImplementedError(
                "Dbrx does not support gptq/awq quantization yet."
            )
        else:
            offset = i * expert_size
            expert_slice = (
                slice_[start + offset : stop + offset]
                .to(dtype=weights.dtype)
                .to(device=weights.device)
            )

        if cls == TensorParallelRowLinear:
            expert_slice = expert_slice.t().contiguous()
            linear = get_linear(expert_slice, None)
            experts.append(cls(linear, weights.process_group))
        else:
            linear = get_linear(expert_slice, None)
            experts.append(cls(linear))

    return experts


class DbrxAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.clip_qkv = config.attn_config.clip_qkv
        self.num_heads = config.n_heads
        self.hidden_size = config.d_model
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_size,
            base=config.attn_config.rope_theta,
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
            config.attn_config.kv_n_heads // weights.process_group.size()
        )

        self.query_key_value = load_attention(config, prefix, weights)

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.out_proj",
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
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        qkv = self.query_key_value(hidden_states)
        if self.clip_qkv is not None:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

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

        reshape_and_cache(kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots)

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attention(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            attn_output = paged_attention(
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


class DbrxNormAttentionNorm(nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.norm_1 = FastLayerNorm.load_no_bias(
            prefix=f"{prefix}.norm_1", weights=weights, eps=1e-5
        )
        self.self_attn = DbrxAttention(
            prefix=f"{prefix}.attn", config=config, weights=weights
        )
        self.norm_2 = FastLayerNorm.load_no_bias(
            prefix=f"{prefix}.norm_2",
            weights=weights,
            eps=1e-5,
        )

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
        normed_hidden_states, res = self.norm_1(hidden_states, residual)

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

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.norm_2(attn_output, res)

        return normed_attn_res_output, attn_res


@torch.jit.script
def select_experts(
    gate_logits: torch.Tensor, top_k: int, moe_normalize_expert_weights: int
):
    # all_probs: (sequence_length, n_experts) and upcast for softmax
    all_probs = torch.nn.functional.softmax(gate_logits, dim=1, dtype=torch.float)
    # weights, selected_experts: (sequence_length, top-k)
    weights, selected_experts = torch.topk(all_probs, top_k, dim=-1)
    if moe_normalize_expert_weights:
        weights = weights / torch.norm(
            weights, p=moe_normalize_expert_weights, dim=-1, keepdim=True
        )
    weights = weights.view(-1)
    selected_experts = selected_experts.view(-1)

    return selected_experts, weights


@torch.jit.script
def round_up(x: torch.Tensor, value: int):
    return torch.div(x + (value - 1), value, rounding_mode="trunc") * value


class BlockSparseMoE(nn.Module):
    def __init__(self, prefix, config: DbrxConfig, weights):
        super().__init__()
        self.moe_normalize_expert_weights = (
            config.ffn_config.moe_normalize_expert_weights
        )
        self.hidden_dim = config.d_model
        self.ffn_dim = config.ffn_config.ffn_hidden_size // weights.process_group.size()
        self.num_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k

        act = config.ffn_config.ffn_act_fn["name"]
        if "gelu" in act:
            self.act = lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        elif "silu" in act:
            self.act = torch.nn.functional.silu
        else:
            self.act = ACT2FN[act]

        # gating
        self.gate = FastLinear.load(
            config, f"{prefix}.router.layer", weights, bias=False
        )

        # merged expert weights, all of size  (n_experts * ffn_dim, hidden_dim)
        w1 = _load_experts(config, f"{prefix}.experts.mlp.w1", weights).view(
            self.num_experts, self.ffn_dim, self.hidden_dim
        )
        v1 = _load_experts(config, f"{prefix}.experts.mlp.v1", weights).view(
            self.num_experts, self.ffn_dim, self.hidden_dim
        )
        self.wv1 = torch.cat([w1, v1], dim=1)
        self.w2 = (
            _load_experts(config, f"{prefix}.experts.mlp.w2", weights)
            .view(self.num_experts, self.ffn_dim, self.hidden_dim)
            .transpose(1, 2)
            .contiguous()
        )

        self.process_group = weights.process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(x)
        out = fused_moe(
            x,
            self.wv1,
            self.w2,
            router_logits,
            self.top_k,
            renormalize=self.moe_normalize_expert_weights,
            inplace=True,
        )

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out.view(*x.shape)


class DenseMoE(nn.Module):
    def __init__(self, prefix, config: DbrxConfig, weights):
        super().__init__()

        self.moe_normalize_expert_weights = (
            config.ffn_config.moe_normalize_expert_weights
        )
        self.hidden_dim = config.d_model
        self.ffn_dim = config.ffn_config.ffn_hidden_size // weights.process_group.size()
        self.num_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k

        act = config.ffn_config.ffn_act_fn["name"]
        if "gelu" in act:
            self.act = lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        elif "silu" in act:
            self.act = torch.nn.functional.silu
        else:
            self.act = ACT2FN[act]

        # gating
        self.gate = FastLinear.load(
            config, f"{prefix}.router.layer", weights, bias=False
        )

        self.w1 = _load_experts_quantized(
            config,
            prefix=f"{prefix}.experts.mlp.w1",
            weights=weights,
            cls=TensorParallelColumnLinear,
        )
        self.w2 = _load_experts_quantized(
            config,
            prefix=f"{prefix}.experts.mlp.w2",
            weights=weights,
            cls=TensorParallelRowLinear,
        )
        self.v1 = _load_experts_quantized(
            config,
            prefix=f"{prefix}.experts.mlp.v1",
            weights=weights,
            cls=TensorParallelColumnLinear,
        )

        self.process_group = weights.process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        # optional reshape
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)
        # all_probs: (sequence_length, n_experts) and upcast for softmax
        weights = torch.nn.functional.softmax(gate_logits, dim=1, dtype=torch.float)

        if self.top_k < self.num_experts:
            _, not_selected_experts = torch.topk(
                weights,
                self.num_experts - self.top_k,
                largest=False,
                sorted=False,
                dim=1,
            )
            # Mask not selected experts
            weights.scatter_(1, not_selected_experts, 0)

        # Re-normalize
        if self.moe_normalize_expert_weights:
            weights = weights / torch.norm(
                weights, p=self.moe_normalize_expert_weights, dim=-1, keepdim=True
            )
        weights = weights.to(x.dtype)

        # Final output tensor
        out = x.new_zeros(x.shape[0], self.hidden_dim)
        for i in range(self.num_experts):
            h = self.act(self.w1[i](x)) * self.v1[i](x)
            h = self.w2[i](h, reduce=False)
            # Add expert output to out with masking
            out += h * weights[:, i].view(-1, 1)

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out


class DbrxLayer(nn.Module):
    def __init__(self, prefix: str, layer_id, config, weights):
        super().__init__()
        prefix = f"{prefix}.blocks.{layer_id}"

        self.attn = DbrxNormAttentionNorm(
            prefix=f"{prefix}.norm_attn_norm", config=config, weights=weights
        )

        moe_cls = BlockSparseMoE if config.quantize is None else DenseMoE
        self.moe = moe_cls(f"{prefix}.ffn", config, weights)

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
        # Self Attention
        attn_output, attn_res = self.attn(
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
        )

        moe_output = self.moe(attn_output)

        return moe_output, attn_res


class DbrxModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.wte", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                DbrxLayer(
                    prefix,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.n_layers)
            ]
        )
        self.norm = FastLayerNorm.load_no_bias(
            prefix=f"{prefix}.norm_f", weights=weights, eps=1e-5
        )

        self.head_size = self.layers[0].attn.self_attn.head_size
        self.num_heads = self.layers[0].attn.self_attn.num_heads
        self.num_key_value_heads = self.layers[0].attn.self_attn.num_key_value_heads

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
        cos, sin = self.layers[0].attn.self_attn.rotary_emb.get_cos_sin(
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


class FlashDbrxForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        if not prefix:
            prefix = "transformer"
        else:
            prefix = f"{prefix}.transformer"

        self.model = DbrxModel(prefix, config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="lm_head",
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
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
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
        return logits, speculative_logits
