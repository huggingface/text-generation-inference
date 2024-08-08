import intel_extension_for_pytorch as ipex
import torch
from text_generation_server.models.flash_causal_lm import BLOCK_SIZE
from text_generation_server.layers.attention import Seqlen
from typing import Optional

SUPPORTS_WINDOWING = False


def attention(
    q,
    k,
    v,
    cu_seqlens,
    max_s,
    softmax_scale,
    window_size_left=-1,
    causal=True,
    softcap: Optional[float] = None,
):
    out = torch.empty_like(q)

    # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
    ipex.llm.functional.varlen_attention(
        q,
        k,
        v,
        out,
        cu_seqlens,
        cu_seqlens,
        max_s,
        max_s,
        0.0,
        softmax_scale,
        False,
        causal,
        False,
        None,
    )

    return out


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
):
    ipex.llm.modules.PagedAttention.reshape_and_cache(
        key, value, key_cache, value_cache, slots
    )


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    seqlen: Seqlen,
    max_s: int,
    softcap: Optional[float] = None,
):
    out = torch.empty_like(query)
    ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
        out,
        query,
        key_cache,
        value_cache,
        kv_head_mapping,
        softmax_scale,
        block_tables,
        seqlen.input_lengths,
        BLOCK_SIZE,
        max_s,
        None,
    )
    return out
