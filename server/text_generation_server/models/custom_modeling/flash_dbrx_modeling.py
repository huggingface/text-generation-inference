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

import numpy as np

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple, Any
from loguru import logger

from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.utils.layers import (
    FastLinear,
    FastLayerNorm,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.utils.log import log_once

HAS_MEGABLOCKS = True
try:
    import stk
    import megablocks.ops as ops
except ImportError:
    logger.warning("Dbrx: megablocks is not installed")
    HAS_MEGABLOCKS = False


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


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


def load_attention(config, prefix, weights):
    if config.n_heads != config.attn_config.kv_n_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.Wqkv",
            weights=weights,
            bias=False,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.d_model % config.n_heads == 0
    assert config.n_heads % weights.process_group.size() == 0

    head_dim = config.d_model // config.n_heads
    world_size = weights.process_group.size()
    rank = weights.process_group.rank()

    q_block_size = config.d_model // world_size
    q_start = rank * q_block_size
    q_stop = (rank + 1) * q_block_size

    kv_block_size = (config.attn_config.kv_n_heads * head_dim) // world_size
    k_offset = config.d_model
    k_start = k_offset + rank * kv_block_size
    k_stop = k_offset + (rank + 1) * kv_block_size

    v_offset = config.d_model + config.attn_config.kv_n_heads * head_dim
    v_start = v_offset + rank * kv_block_size
    v_stop = v_offset + (rank + 1) * kv_block_size

    if config.quantize in ["gptq", "awq"]:
        try:
            qweight_slice = weights._get_slice(f"{prefix}.qweight")
            q_qweight = qweight_slice[:, q_start:q_stop]
            k_qweight = qweight_slice[:, k_start:k_stop]
            v_qweight = qweight_slice[:, v_start:v_stop]

            qweight = torch.cat([q_qweight, k_qweight, v_qweight], dim=1)
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load `{config.quantize}` weight, make sure the model is already quantized"
            )

        qzeros_slice = weights._get_slice(f"{prefix}.qzeros")
        q_qzeros = qzeros_slice[:, q_start:q_stop]
        k_qzeros = qzeros_slice[:, k_start:k_stop]
        v_qzeros = qzeros_slice[:, v_start:v_stop]

        qzeros = torch.cat([q_qzeros, k_qzeros, v_qzeros], dim=1)

        scales_slice = weights._get_slice(f"{prefix}.scales")
        q_scales = scales_slice[:, q_start:q_stop]
        k_scales = scales_slice[:, k_start:k_stop]
        v_scales = scales_slice[:, v_start:v_stop]

        scales = torch.cat([q_scales, k_scales, v_scales], dim=1)

        bits, groupsize, desc_act, quant_method = weights._get_gptq_params()

        from text_generation_server.utils.layers import HAS_EXLLAMA

        use_exllama = (
            bits == 4 and HAS_EXLLAMA and config.quantize == "gptq" and not desc_act
        )

        if config.quantize == "gptq" and quant_method == "gptq":
            g_idx_slice = weights._get_slice(f"{prefix}.g_idx")
            q_g_idx = g_idx_slice[:, q_start:q_stop]
            k_g_idx = g_idx_slice[:, k_start:k_stop]
            v_g_idx = g_idx_slice[:, v_start:v_stop]

            w = [q_g_idx, k_g_idx, v_g_idx]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]
        elif config.quantize == "gptq" and quant_method == "awq":
            log_once(
                logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
            )
            from text_generation_server.utils.awq.conversion_utils import (
                fast_awq_to_gptq,
            )

            qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
            if use_exllama:
                g_idx = None
            else:
                g_idx = (
                    torch.arange(qweight.shape[0] * (32 // bits), device=qweight.device)
                    // groupsize
                ).to(dtype=torch.int32)
        else:
            g_idx = None

        weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)
    else:
        qkv_slice = weights._get_slice(f"{prefix}.Wqkv.weight")
        q = qkv_slice[q_start:q_stop]
        k = qkv_slice[k_start:k_stop]
        v = qkv_slice[v_start:v_stop]

        weight = torch.cat([q, k, v], dim=0)
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
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
            linear = get_linear(expert_slice, None, config.quantize)
            experts.append(cls(linear, weights.process_group))
        else:
            linear = get_linear(expert_slice, None, config.quantize)
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

        paged_attention.reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
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
    """
    Built on the paper and library Megablocks as described in
    https://arxiv.org/abs/2211.15841. This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

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
        self.w1 = _load_experts(config, f"{prefix}.experts.mlp.w1", weights)
        self.w2 = _load_experts(config, f"{prefix}.experts.mlp.w2", weights)
        self.v1 = _load_experts(config, f"{prefix}.experts.mlp.v1", weights)

        self.offsets = None
        self.offsets_block_rows = 0

        self.process_group = weights.process_group

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.blocking = 128
        self.quantize_scatter_num_bits = -1

    def topology(self, x: torch.Tensor, padded_bins: torch.Tensor):
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        assert self.ffn_dim % self.blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_dim // self.blocking
        if self.offsets is None or block_rows > self.offsets_block_rows:
            self.offsets = torch.arange(
                0,
                block_rows * blocks_per_row + 1,
                blocks_per_row,
                dtype=torch.int32,
                device=x.device,
            )
            self.offsets_block_rows = block_rows
            offsets = self.offsets
        else:
            offsets = self.offsets[: block_rows + 1]

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(
            padded_bins, self.blocking, block_rows, blocks_per_row
        )

        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=x.dtype,
            device="meta",
        )
        shape = (padded_tokens, self.ffn_dim * self.num_experts)
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        return stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            False,
            False,
            False,
        )

    def indices_and_padded_bins(self, selected_experts: torch.Tensor):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        # selected_experts = selected_experts.int()

        # returns bin_ids == num of experts for this sequence ? == unique selected experts?
        # and indices == how to sort tokens?
        bin_ids, indices = ops.sort(selected_experts, self.sort_end_bit)
        # bin_ids => [0, 0, 0, 2, 2, ...] => [num_tokens * top_k]
        # indices => [14, 32, 33, ...] => [num_tokens * top_k]

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(selected_experts, self.num_experts)
        # tokens_per_expert => [3, 0, 2, ...] => [num_experts]

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.

        # List of size num_experts
        padded_tokens_per_expert = round_up(tokens_per_expert, self.blocking)
        # padded_tokens_per_expert => [128, O, 128, ...]

        # Cumulative selected experts per token
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)
        # padded_bins => [128, 128, 256, ...]

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        # bins => [3, 3, 5, ...]

        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    def sparse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        # optional reshape
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)
        selected_experts, weights = select_experts(
            gate_logits, self.top_k, self.moe_normalize_expert_weights
        )

        (
            indices,
            bin_ids,
            bins,
            padded_bins,
            _,
        ) = self.indices_and_padded_bins(selected_experts)

        # Permute tokens and pad to prepare expert computation
        # (top_k * sequence_length + padding, model_dim)
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, self.top_k)

        # Create the sparse matrix topology
        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and v1,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = stk.Matrix(
            topo.size(),
            self.act(stk.ops.sdd(x, self.w1.t(), topo).data)
            * stk.ops.sdd(x, self.v1.t(), topo).data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        # Then Sparse x Dense -> Dense for w2
        # (top_k * sequence_length + padding, model_dim)
        x = stk.ops.dsd(x, self.w2)

        # Permute back and remove padding
        # (sequence_length, model_dim)
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            self.top_k,
            self.quantize_scatter_num_bits,
        ).view(*input_shape)

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(x, group=self.process_group)

        return x.view(*input_shape)

    def dense_forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Expand to [num_experts, sequence_length, model_dim]
        x = x.view(1, -1, input_shape[-1]).expand(self.num_experts, -1, input_shape[-1])

        # Permute to [num_experts, model_dim, ffn_dim]
        w1 = self.w1.view(self.num_experts, self.ffn_dim, self.hidden_dim).permute(
            0, 2, 1
        )
        v1 = self.v1.view(self.num_experts, self.ffn_dim, self.hidden_dim).permute(
            0, 2, 1
        )

        inter = self.act(torch.bmm(x, w1)) * torch.bmm(x, v1)

        out = torch.bmm(
            inter, self.w2.view(self.num_experts, self.ffn_dim, self.hidden_dim)
        )
        # Mask not selected experts
        out *= weights.t().view(self.num_experts, -1, 1)

        # Sum experts
        out = out.sum(0)

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) > 256 and HAS_MEGABLOCKS:
            return self.sparse_forward(x)
        # This is faster when there is not a lot of tokens
        return self.dense_forward(x)


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
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.blocks.{layer_id}"

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
    def __init__(self, config, weights):
        super().__init__()

        self.embed_tokens = TensorParallelEmbedding(
            prefix="transformer.wte", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                DbrxLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.n_layers)
            ]
        )
        self.norm = FastLayerNorm.load_no_bias(
            prefix="transformer.norm_f", weights=weights, eps=1e-5
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
    def __init__(self, config, weights):
        super().__init__()

        self.model = DbrxModel(config, weights)
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
        return logits, speculative_logits
