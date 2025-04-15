import intel_extension_for_pytorch as ipex
import torch
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from text_generation_server.layers.attention import Seqlen
from typing import Optional
from text_generation_server.models.globals import (
    ATTENTION,
    BLOCK_SIZE,
)

if ATTENTION == "flashdecoding-ipex":
    SUPPORTS_WINDOWING = True
else:
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

    out = torch.empty_like(query)
    kv_cache_dtype = "auto"
    if kv_cache.key.dtype == torch.float8_e5m2:
        kv_cache_dtype = "fp8_e5m2"
    if kv_cache.key.dtype == torch.float8_e4m3fn:
        kv_cache_dtype = "fp8_e4m3"

    # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
    if ATTENTION == "flashdecoding-ipex":
        window_size_right = -1 if window_size_left == -1 else 0
        if softcap is None:
            softcap = -1.0
        ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            out,
            query.contiguous() if query.device.type == "xpu" else query,
            kv_cache.key,
            kv_cache.value,
            seqlen.cu_seqlen_q,
            seqlen.cu_seqlen_k,
            seqlen.max_q,
            seqlen.max_k,
            softmax_scale,
            causal,
            block_tables,
            None,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=kv_scales.key_scale_cpu,
            v_scale=kv_scales.value_scale_cpu,
            softcap=softcap,
        )
    else:
        if softcap is not None:
            raise NotImplementedError(
                "softcap is not available in IPEX paged attention"
            )
        ipex.llm.functional.varlen_attention(
            query.contiguous() if query.device.type == "xpu" else query,
            key.contiguous() if key.device.type == "xpu" else key,
            value.contiguous() if value.device.type == "xpu" else value,
            out,
            seqlen.cu_seqlen_q,
            seqlen.cu_seqlen_q,
            seqlen.max_q,
            seqlen.max_q,
            0.0,
            softmax_scale,
            False,
            causal,
            False,
            None,
        )

    return out


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
    out = torch.empty_like(query)
    kv_cache_dtype = "auto"
    if kv_cache.key.dtype == torch.float8_e5m2:
        kv_cache_dtype = "fp8_e5m2"
    if kv_cache.key.dtype == torch.float8_e4m3fn:
        kv_cache_dtype = "fp8_e4m3"
    if ATTENTION == "flashdecoding-ipex":
        window_size_right = -1 if window_size_left == -1 else 0
        if softcap is None:
            softcap = -1.0
        ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            out,
            query.contiguous() if query.device.type == "xpu" else query,
            kv_cache.key,
            kv_cache.value,
            seqlen.cu_seqlen_q,
            seqlen.cu_seqlen_k,
            seqlen.max_q,
            seqlen.max_k,
            softmax_scale,
            True,
            block_tables,
            None,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=kv_scales.key_scale_cpu,
            v_scale=kv_scales.value_scale_cpu,
            softcap=softcap,
        )
    else:
        input_lengths = seqlen.input_lengths + seqlen.cache_lengths
        if softcap is not None:
            raise NotImplementedError(
                "softcap is not available in IPEX paged attention"
            )
        ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
            out,
            query,
            kv_cache.key,
            kv_cache.value,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            BLOCK_SIZE,
            max_s,
            None,
            k_scale=kv_scales.key_scale_cpu,
            v_scale=kv_scales.value_scale_cpu,
        )
    return out


__all__ = [
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
]
