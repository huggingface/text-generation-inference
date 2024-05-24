import torch
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models.globals import FLASH_DECODING

_PARTITION_SIZE = 512

if SYSTEM == "xpu":
    import intel_extension_for_pytorch as ipex
else:
    try:
        from vllm._C import cache_ops
        from vllm._C import ops
    except Exception as e:
        raise ImportError(
            f"Could not import vllm paged attention. Make sure your installation is correct. Complete error: {e}"
        )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
):
    if SYSTEM == "xpu":
        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache, slots
        )
    else:
        if FLASH_DECODING:
            shape = key_cache.shape
            key_cache.view(-1, shape[-2], shape[-1])[slots] = key
            value_cache.view(-1, shape[-2], shape[-1])[slots] = value
        else:
            cache_ops.reshape_and_cache(
                key, value, key_cache, value_cache, slots, "auto", 1.0
            )


def attention(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    input_lengths: torch.Tensor,
    max_s: int,
):
    # Adapted from: https://github.com/vllm-project/vllm/blob/f8a1e39fae05ca610be8d5a78be9d40f5274e5fc/vllm/model_executor/layers/attention.py
    # Copyright 2023 The vLLM team. All rights
    # reserved.
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
    #

    # value_cache => [num_blocks, num_heads, head_size, block_size]
    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (max_s + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    if SYSTEM == "xpu":
        query = query.contiguous()
        return ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
            out,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
        )

    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    if FLASH_DECODING:
        cu_seqlen_q = torch.arange(
            input_lengths.shape[0] + 1, device=query.device, dtype=torch.int32
        )
        cu_seqlen_k = torch.cat(
            [
                torch.zeros(
                    (1,), device=input_lengths.device, dtype=input_lengths.dtype
                ),
                input_lengths.cumsum(dim=-1),
            ]
        ).to(dtype=torch.int32)
        max_q = 1
        max_k = max_s
        import flash_attn_2_cuda

        flash_attn_2_cuda.varlen_fwd(
            query,
            key_cache,
            value_cache,
            out,
            cu_seqlen_q,
            cu_seqlen_k,
            None,
            block_tables,
            None,
            max_q,
            max_k,
            0.0,
            softmax_scale,
            False,
            True,
            -1,
            0,
            False,
            None,
        )
    else:
        from vllm._C import ops

        use_v1 = max_s <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512
        )
        if use_v1:
            ops.paged_attention_v1(
                out,
                query,
                key_cache,
                value_cache,
                kv_head_mapping,
                softmax_scale,
                block_tables,
                input_lengths,
                block_size,
                max_s,
                None,
                "auto",
                1.0,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=out.dtype,
                device=out.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=out.device,
            )
            max_logits = torch.empty_like(exp_sums)

            ops.paged_attention_v2(
                out,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                kv_head_mapping,
                softmax_scale,
                block_tables,
                input_lengths,
                block_size,
                max_s,
                None,
                "auto",
                1.0,
            )
