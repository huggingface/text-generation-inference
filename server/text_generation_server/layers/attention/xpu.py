import intel_extension_for_pytorch as ipex
import torch

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
    if window_size_left != -1:
        raise ValueError(
            f"XPU version of Flash Attention does not support window attention (window_size_left != -1, got window_size_left={window_size_left})."
        )
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
    query = query.contiguous()
    block_size = value_cache.shape[3]
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
