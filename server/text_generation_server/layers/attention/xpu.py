import intel_extension_for_pytorch as ipex
import torch
from text_generation_server.models.flash_causal_lm import BLOCK_SIZE

SUPPORTS_WINDOWING = False


def attention(
    q,
    k,
    v,
    out,
    cu_seqlens,
    max_s,
    softmax_scale,
    window_size_left=-1,
):
    # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
    return ipex.llm.functional.varlen_attention(
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
        True,
        False,
        None,
    )


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
    return ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
        out,
        query,
        key_cache,
        value_cache,
        kv_head_mapping,
        softmax_scale,
        block_tables,
        input_lengths,
        BLOCK_SIZE,
        max_s,
        None,
    )
