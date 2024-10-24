import os
from typing import Optional
import torch
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.layers.attention import Seqlen
from text_generation_server.utils.log import log_master
from loguru import logger

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5

_PARTITION_SIZE_V1V2 = 512
_PARTITION_SIZE_CUSTOM = 256

use_triton = os.getenv("ROCM_USE_FLASH_ATTN_V2_TRITON", "").lower() in {"true", "1"}
ENGINE = "triton" if use_triton else "ck"

use_rocm_custom_paged_attn = os.getenv("ROCM_USE_CUSTOM_PAGED_ATTN", "1") != "0"
try:
    if use_rocm_custom_paged_attn:
        from vllm._custom_C import paged_attention_custom
except ImportError as e:
    log_master(
        logger.info,
        f"Custom Paged Attention not available. Complete error: {e}",
    )
    use_rocm_custom_paged_attn = False


def paged_attention(
    query: torch.Tensor,
    kv_cache: KVCache,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    seqlen: Seqlen,
    max_s: int,
    *,
    kv_scales: KVScales,
    softcap: Optional[float] = None,
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

    if softcap is not None:
        raise RuntimeError("Paged attention doesn't support softcapping")

    # value_cache => [num_blocks, num_heads, head_size, block_size]
    block_size = kv_cache.value.shape[3]
    num_seqs, num_heads, head_size = query.shape

    num_kv_heads = kv_cache.key.shape[1]
    gqa_ratio = num_heads // num_kv_heads
    use_custom = (
        use_rocm_custom_paged_attn
        and (query.dtype == torch.half or query.dtype == torch.bfloat16)
        and (head_size == 128 or head_size == 64)
        and (block_size == 16 or block_size == 32)
        and (gqa_ratio >= 1 and gqa_ratio <= 16)
        and max_s <= 32768
    )

    if not use_custom:
        _PARTITION_SIZE = _PARTITION_SIZE_V1V2
    else:
        _PARTITION_SIZE = _PARTITION_SIZE_CUSTOM

    max_num_partitions = (max_s + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    input_lengths = seqlen.input_lengths + seqlen.cache_lengths

    out = torch.empty_like(query)

    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    import vllm._custom_ops as ops

    use_v1 = (
        max_s <= 8192
        and (max_num_partitions == 1 or num_seqs * num_heads > 512)
        and not use_custom
    )
    if use_v1:
        ops.paged_attention_v1(
            out,
            query,
            kv_cache.key,
            kv_cache.value,
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

        if not use_custom:
            ops.paged_attention_v2(
                out,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                kv_cache.key,
                kv_cache.value,
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
            paged_attention_custom(
                out,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                kv_cache.key,
                kv_cache.value,
                num_kv_heads,
                softmax_scale,
                block_tables,
                input_lengths,
                block_size,
                max_s,
                None,
                "auto",
            )

    return out


if ENGINE != "triton":
    try:
        import flash_attn_2_cuda

        log_master(
            logger.info,
            "ROCm: using Flash Attention 2 Composable Kernel implementation.",
        )
    except ImportError as e:
        if major >= 8:
            architecture_suffix = f"-{SYSTEM}"
            raise ImportError(
                "Flash Attention V2 is not installed.\n"
                "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
                f"or install flash attention v2 with `cd server && make install install-flash-attention-v2{architecture_suffix}`"
            )
        elif is_sm75:
            raise ImportError(
                "Flash Attention is not installed.\n"
                "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
                "or install flash attention with `cd server && make install install-flash-attention`"
            ) from e
        else:
            for idx in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(idx)
                if "MI210" not in name and "MI250" not in name:
                    raise ImportError(
                        f"AMD GPU {torch.cuda.get_device_name(idx)} does not support flash-attention"
                    )
            raise ImportError(
                f"AMD GPU with ROCm capability {major} {minor} is not supported"
            ) from e


SUPPORTS_WINDOWING = False


def attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: KVCache,
    kv_scales: KVScales,
    seqlen: Seqlen,
    block_tables: torch.Tensor,
    softmax_scale: float,
    window_size_left: int = -1,
    causal: bool = True,
    softcap: Optional[float] = None,
):
    if ENGINE == "ck":
        if window_size_left <= 0 and window_size_left != -1:
            raise ValueError("`window_size_left` must be > 0 or -1")

        out = torch.empty_like(query)

        if softcap is None:
            softcap = 0.0

        # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
        return flash_attn_2_cuda.varlen_fwd(
            query,
            key,
            value,
            out,
            seqlen.cu_seqlen_q,
            seqlen.cu_seqlen_q,
            None,
            None,
            None,
            None,
            seqlen.max_q,
            seqlen.max_k,
            0.0,
            softmax_scale,
            False,
            causal,
            window_size_left,
            0,
            softcap,
            False,
            None,
        )[0]

    elif ENGINE == "triton":
        from .flash_attn_triton import triton_attention

        if softcap is not None:
            raise NotImplementedError("softcap is only available with CK flash attn")

        out = torch.empty_like(query)

        # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
        output, _ = triton_attention(
            query,
            key,
            value,
            out,
            seqlen.cu_seqlen_q,
            seqlen.cu_seqlen_q,
            seqlen.max_q,
            seqlen.max_k,
            causal,
            softmax_scale,
        )
        return output

    else:
        raise RuntimeError(f"Unknown attention engine {ENGINE}")


__all__ = [
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
]
