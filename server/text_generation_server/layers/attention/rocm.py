import os
from typing import Optional
import torch
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.layers.attention import Seqlen
from text_generation_server.utils.log import log_master
from text_generation_server.models.globals import (
    ATTENTION,
    BLOCK_SIZE,
)
from loguru import logger
import vllm._custom_ops as ops

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5

_PARTITION_SIZE_V1V2 = 1024
_PARTITION_SIZE_CUSTOM = 256

_GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
_ON_MI250_MI300 = any(
    arch in _GPU_ARCH for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]
)

use_triton = os.getenv("ROCM_USE_FLASH_ATTN_V2_TRITON", "").lower() in {"true", "1"}
ENGINE = "triton" if use_triton else "ck"

use_rocm_custom_paged_attn = os.getenv("ROCM_USE_CUSTOM_PAGED_ATTN", "1") != "0"


def _use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
) -> bool:
    # rocm custom page attention not support on navi (gfx1*)
    return (
        use_rocm_custom_paged_attn
        and _ON_MI250_MI300
        and (qtype == torch.half or qtype == torch.bfloat16)
        and (head_size == 64 or head_size == 128)
        and (block_size == 16 or block_size == 32)
        and (gqa_ratio >= 1 and gqa_ratio <= 16)
        and max_seq_len <= 131072
    )


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
    window_size_left: Optional[int] = -1,
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

    if ATTENTION == "flashdecoding":
        max_q = 1
        max_k = max_s
        import flash_attn_2_cuda

        window_size_right = -1 if window_size_left == -1 else 0

        if softcap is None:
            softcap = 0.0
        out = flash_attn_2_cuda.varlen_fwd(
            query,
            kv_cache.key,
            kv_cache.value,
            None,
            seqlen.cu_seqlen_q,
            seqlen.cu_seqlen_k,
            None,  # pad_k
            None,
            block_tables,
            None,
            max_q,
            max_k,
            0.0,  # dropout
            softmax_scale,
            False,  # zero_tensors
            True,  # causal
            window_size_left,  # Window_left
            window_size_right,  # Window right
            softcap,
            False,  # return softmax
            None,  # generator
        )
        return out[0]

    if softcap is not None:
        raise RuntimeError("Paged attention doesn't support softcapping")

    # value_cache => [num_blocks, num_heads, head_size, block_size]
    # block_size = kv_cache.value.shape[3]
    block_size = BLOCK_SIZE
    num_seqs, num_heads, head_size = query.shape

    num_kv_heads = kv_cache.key.shape[1]
    gqa_ratio = num_heads // num_kv_heads
    use_custom = _use_rocm_custom_paged_attention(
        query.dtype, head_size, block_size, gqa_ratio, max_s
    )

    if not use_custom:
        _PARTITION_SIZE = _PARTITION_SIZE_V1V2
    else:
        _PARTITION_SIZE = _PARTITION_SIZE_CUSTOM

    max_num_partitions = (max_s + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    input_lengths = seqlen.input_lengths + seqlen.cache_lengths

    out = torch.empty_like(query)

    if kv_cache.dtype == torch.float8_e4m3fn:
        key = kv_cache.key.view(torch.uint8)
        value = kv_cache.value.view(torch.uint8)
        kv_cache_dtype = "fp8"
    else:
        key = kv_cache.key
        value = kv_cache.value
        kv_cache_dtype = "auto"

    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    use_v1 = (
        max_s <= 8192
        and (max_num_partitions == 1 or num_seqs * num_heads > 512)
        and not use_custom
    )
    if use_v1:
        ops.paged_attention_v1(
            out,
            query,
            key,
            value,
            num_kv_heads,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
            kv_cache_dtype,
            kv_scales.key_scale_cpu,
            kv_scales.value_scale_cpu,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.zeros(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        exp_sums = torch.zeros(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=out.device,
        )
        max_logits = torch.zeros_like(exp_sums)

        if not use_custom:
            ops.paged_attention_v2(
                out,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key,
                value,
                num_kv_heads,
                softmax_scale,
                block_tables,
                input_lengths,
                block_size,
                max_s,
                None,
                kv_cache_dtype,
                kv_scales.key_scale_cpu,
                kv_scales.value_scale_cpu,
            )
        else:
            ops.paged_attention_rocm(
                out,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key,
                value,
                num_kv_heads,
                softmax_scale,
                block_tables,
                input_lengths,
                block_size,
                max_s,
                None,
                kv_cache_dtype,
                kv_scales.key_scale_cpu,
                kv_scales.value_scale_cpu,
                None,
                _PARTITION_SIZE,
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
            # flashdecoding: pass the KV caches, paged: pass the KV.
            kv_cache.key if ATTENTION == "flashdecoding" else key,
            kv_cache.value if ATTENTION == "flashdecoding" else value,
            out,
            seqlen.cu_seqlen_q,
            seqlen.cu_seqlen_k,
            None,
            None,
            block_tables if ATTENTION == "flashdecoding" else None,
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
