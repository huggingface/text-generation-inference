import intel_extension_for_pytorch as ipex
import torch
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from text_generation_server.models.flash_causal_lm import BLOCK_SIZE
from text_generation_server.layers.attention import Seqlen
from typing import Optional

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
    if softcap is not None:
        raise NotImplementedError("softcap is not available in IPEX")

    out = torch.empty_like(query)

    # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
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
):
    if softcap is not None:
        raise NotImplementedError("softcap is not available in IPEX")

    out = torch.empty_like(query)
    input_lengths = seqlen.input_lengths + seqlen.cache_lengths
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
    )
    return out


__all__ = [
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
]
