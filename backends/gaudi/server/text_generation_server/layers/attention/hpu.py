import torch
from text_generation_server.layers.attention import Seqlen, HPUPagedAttentionMetadata
from typing import Optional
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from vllm_hpu_extension import ops
from vllm_hpu_extension.utils import Matmul
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from vllm_hpu_extension.utils import ModuleFusedSDPA
import os

SUPPORTS_WINDOWING = False


def fetch_from_cache(cache, blocks):
    if os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() == "true":
        return cache[: blocks.size(0)]
    else:
        return cache.index_select(0, blocks)


def attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: KVCache,
    kv_scales: KVScales,
    seqlen: Seqlen,
    softmax_scale: float,
    window_size_left: int = -1,
    causal: bool = True,
    softcap: Optional[float] = None,
):
    fsdpa_op = ModuleFusedSDPA(FusedSDPA)
    bs = seqlen.input_lengths.shape[0]
    _, head_num, head_size = query.shape
    _, kv_head_num, head_size = key.shape
    query = query.view(bs, -1, head_num, head_size).transpose(1, 2)
    key = key.view(bs, -1, kv_head_num, head_size).transpose(1, 2)
    value = value.view(bs, -1, kv_head_num, head_size).transpose(1, 2)
    attn_output = fsdpa_op(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale,
        softmax_mode="None",
        recompute_mode=None,
        valid_sequence_lengths=seqlen.input_lengths,
        padding_side="left",
    )
    attn_output = attn_output.transpose(1, 2).squeeze(0).contiguous()
    return attn_output


def paged_attention(
    query: torch.Tensor,
    kv_cache: KVCache,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    seqlen: Seqlen,
    *,
    kv_scales: KVScales,
    softcap: Optional[float] = None,
    hpu_attention_meta: HPUPagedAttentionMetadata,
):
    batch_size, head_num, head_size = query.shape
    output = ops.flat_pa(
        query=query.view(batch_size, 1, head_num * head_size),
        key_cache=kv_cache.key,
        value_cache=kv_cache.value,
        block_list=hpu_attention_meta.block_list,
        block_mapping=hpu_attention_meta.block_mapping,
        block_bias=hpu_attention_meta.attn_bias,
        block_scales=hpu_attention_meta.block_scales,
        block_groups=hpu_attention_meta.block_groups,
        scale=softmax_scale,
        matmul_qk_op=Matmul(),
        matmul_av_op=Matmul(),
        batch2block_matmul_op=Matmul(),
        block2batch_matmul_op=Matmul(),
        keys_fetch_func=fetch_from_cache,
        values_fetch_func=fetch_from_cache,
    )
    # Reshape the output tensor.
    return output.view(batch_size, head_num, head_size)


__all__ = [
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
]
