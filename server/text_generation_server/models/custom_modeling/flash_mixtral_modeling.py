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

import numpy as np

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple
from loguru import logger

from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.utils.layers import (
    FastLinear,
    FastRMSNorm,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorParallelHead,
    get_linear,
)

HAS_MEGABLOCKS = True
try:
    import stk
    import megablocks.ops as ops
except ImportError:
    logger.warning("Mixtral: megablocks is not installed")
    HAS_MEGABLOCKS = False


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


def load_attention(config, prefix, weights):
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

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
    )


def _load_experts(config, prefix, mat, weights):
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
    ):
        super().__init__()
        self.max_past = (
            config.sliding_window if config.sliding_window is not None else -1
        )
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
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
        block_tables,
        slots,
        input_lengths,
        max_s,
        prefill_cache_indices,
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

        if prefill_cache_indices is not None:
            kv_to_cache = kv[prefill_cache_indices]
        else:
            kv_to_cache = kv

        paged_attention.reshape_and_cache(
            kv_to_cache[:, 0], kv_to_cache[:, 1], kv_cache[0], kv_cache[1], slots
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
                window_size_left=self.max_past,
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

    def __init__(self, prefix, config: MixtralConfig, weights):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size // weights.process_group.size()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        act = config.hidden_act
        if "gelu" in act:
            self.act = lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        elif "silu" in act:
            self.act = torch.nn.functional.silu
        else:
            self.act = ACT2FN[act]

        # gating
        self.gate = FastLinear.load(config, f"{prefix}.gate", weights, bias=False)

        # merged expert weights, all of size  (n_experts * ffn_dim, hidden_dim)
        self.w1 = _load_experts(config, f"{prefix}.experts", "w1", weights)
        self.w2 = _load_experts(config, f"{prefix}.experts", "w2", weights)
        self.w3 = _load_experts(config, f"{prefix}.experts", "w3", weights)

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
        selected_experts, weights = select_experts(gate_logits, self.top_k)

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
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = stk.Matrix(
            topo.size(),
            self.act(stk.ops.sdd(x, self.w1.t(), topo).data)
            * stk.ops.sdd(x, self.w3.t(), topo).data,
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
        all_probs = torch.nn.functional.softmax(gate_logits, dim=1, dtype=torch.float)

        if self.top_k < self.num_experts:
            _, not_selected_experts = torch.topk(
                all_probs,
                self.num_experts - self.top_k,
                largest=False,
                sorted=False,
                dim=1,
            )
            # Mask not selected experts
            all_probs.scatter_(1, not_selected_experts, 0)

        # Re-normalize
        weights = all_probs / all_probs.sum(dim=1, keepdim=True)

        # Expand to [num_experts, sequence_length, model_dim]
        x = x.view(1, -1, input_shape[-1]).expand(self.num_experts, -1, input_shape[-1])

        # Permute to [num_experts, model_dim, ffn_dim]
        w1 = self.w1.view(self.num_experts, self.ffn_dim, self.hidden_dim).permute(
            0, 2, 1
        )
        w3 = self.w3.view(self.num_experts, self.ffn_dim, self.hidden_dim).permute(
            0, 2, 1
        )

        inter = self.act(torch.bmm(x, w1)) * torch.bmm(x, w3)

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
    def __init__(self, prefix, config: MixtralConfig, weights):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size // weights.process_group.size()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        act = config.hidden_act
        if "gelu" in act:
            self.act = lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        elif "silu" in act:
            self.act = torch.nn.functional.silu
        else:
            self.act = ACT2FN[act]

        # gating
        self.gate = FastLinear.load(config, f"{prefix}.gate", weights, bias=False)

        self.w1 = [
            TensorParallelColumnLinear.load(
                config, prefix=f"{prefix}.experts.{i}.w1", weights=weights, bias=False
            )
            for i in range(self.num_experts)
        ]
        self.w3 = [
            TensorParallelColumnLinear.load(
                config, prefix=f"{prefix}.experts.{i}.w3", weights=weights, bias=False
            )
            for i in range(self.num_experts)
        ]
        self.w2 = [
            TensorParallelRowLinear.load(
                config, prefix=f"{prefix}.experts.{i}.w2", weights=weights, bias=False
            )
            for i in range(self.num_experts)
        ]

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
        all_probs = torch.nn.functional.softmax(gate_logits, dim=1, dtype=torch.float)

        if self.top_k < self.num_experts:
            _, not_selected_experts = torch.topk(
                all_probs,
                self.num_experts - self.top_k,
                largest=False,
                sorted=False,
                dim=1,
            )
            # Mask not selected experts
            all_probs.scatter_(1, not_selected_experts, 0)

        # Re-normalize
        weights = all_probs / all_probs.sum(dim=1, keepdim=True)

        # Final output tensor
        out = x.new_zeros(x.shape[0], self.hidden_dim)
        for i in range(self.num_experts):
            h = self.act(self.w1[i](x)) * self.w3[i](x)
            h = self.w2[i](h, reduce=False)
            # Add expert output to out with masking
            out += h * weights[:, i].view(-1, 1)

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out


class MixtralLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"

        self.self_attn = MixtralAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )

        moe_cls = BlockSparseMoE if config.quantize is None else DenseMoE
        self.moe = moe_cls(f"{prefix}.block_sparse_moe", config, weights)

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
        block_tables,
        slots,
        input_lengths,
        max_s,
        prefill_cache_indices,
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
            prefill_cache_indices,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        moe_output = self.moe(normed_attn_res_output)

        return moe_output, attn_res


class MixtralModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                MixtralLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
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
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        true_max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, true_max_s, hidden_states.dtype
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
                prefill_cache_indices,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashMixtralForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = MixtralModel(config, weights)
        self.lm_head = TensorParallelHead.load(
            config,
            prefix="lm_head",
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
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        true_max_s = max_s
        if prefill_cache_indices is not None:
            # Slots also need to be sliced as it has the same size as the whole kv tensor
            slots = slots[prefill_cache_indices]
        elif self.max_past is not None:
            # Clamp in decode mode as paged attention requires clamped values whereas the flash attention
            # kernel requires the true values
            input_lengths = torch.clamp(input_lengths, max=self.max_past_tensor)

        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            true_max_s,
            prefill_cache_indices,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits
