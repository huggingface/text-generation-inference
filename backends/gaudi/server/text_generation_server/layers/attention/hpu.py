import torch
from text_generation_server.layers.attention import Seqlen, HPUPagedAttentionMetadata
from typing import Optional
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from vllm_hpu_extension import ops
from vllm_hpu_extension.utils import Matmul
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from vllm_hpu_extension.utils import ModuleFusedSDPA
import os
from text_generation_server.models.globals import BLOCK_SIZE
import math

SUPPORTS_WINDOWING = False


class FP8Matmul(torch.nn.Module):

    def __init__(self, scale_other):
        super().__init__()
        self.scale_input = torch.tensor(1.0, dtype=torch.bfloat16, device="hpu")
        self.scale_other = scale_other

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(
            x, scale, False, False, torch.float8_e4m3fn
        )[0]

    def matmul_fp8(
        self, x, other, out_dtype, scale_input_inv=None, scale_other_inv=None
    ):
        return torch.ops.hpu.fp8_gemm_v2(
            A=x,
            trans_A=False,
            B=other,
            trans_B=False,
            D=None,
            out_dtype=out_dtype,
            A_scale_inv=scale_input_inv,
            B_scale_inv=scale_other_inv,
            bias=None,
            accumulate=False,
        )

    def forward(self, input, other):
        qinput = self.quant_input(input, self.scale_input)
        qother = self.quant_input(other, self.scale_other)
        output = self.matmul_fp8(
            qinput,
            qother,
            out_dtype=torch.bfloat16,
            scale_input_inv=1.0 / self.scale_input,
            scale_other_inv=1.0 / self.scale_other,
        )
        return output


class FetchFromCache(torch.nn.Module):

    def __init__(self, scale_inv):
        super().__init__()
        self.scale_inv = scale_inv

    def forward(self, cache, blocks):
        if os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() == "true":
            out = cache[: blocks.size(0)]
        else:
            out = cache.index_select(0, blocks)
        if out.dtype == torch.float8_e4m3fn:
            out = torch.ops.hpu.cast_from_fp8(out, self.scale_inv, torch.bfloat16)
        return out


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
        attn_mask=seqlen.attn_mask if window_size_left != -1 else None,
        dropout_p=0.0,
        is_causal=causal if window_size_left == -1 else False,
        scale=softmax_scale,
        softmax_mode="None",
        recompute_mode=None,
        valid_sequence_lengths=seqlen.input_lengths if window_size_left == -1 else None,
        padding_side="left",
    )
    attn_output = attn_output.transpose(1, 2).squeeze(0).contiguous()
    return attn_output


def set_block_mapping(hpu_attention_meta: HPUPagedAttentionMetadata, batch_size):
    block_mapping = torch.nn.functional.one_hot(
        hpu_attention_meta.block_groups, num_classes=batch_size
    )
    dtype = hpu_attention_meta.block_usage.dtype
    device = hpu_attention_meta.block_usage.device
    mask = torch.arange(0, BLOCK_SIZE, device=device, dtype=torch.int32).unsqueeze(0)
    mask = mask >= hpu_attention_meta.block_usage.unsqueeze(-1)
    attn_bias = torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf)
    hpu_attention_meta = hpu_attention_meta._replace(
        attn_bias=attn_bias, block_mapping=block_mapping.to(dtype)
    )
    if hpu_attention_meta.block_groups_in_window is not None:
        block_mapping = torch.nn.functional.one_hot(
            hpu_attention_meta.block_groups_in_window, num_classes=batch_size
        )
        attn_bias = torch.log(hpu_attention_meta.slots_in_window_mask.float())
        hpu_attention_meta = hpu_attention_meta._replace(
            attn_bias_in_window=attn_bias,
            block_mapping_in_window=block_mapping.to(dtype),
        )
    return hpu_attention_meta


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
    window_size_left: int = -1,
):
    batch_size, head_num, head_size = query.shape
    fp8_kv = kv_cache.dtype == torch.float8_e4m3fn
    output = ops.flat_pa(
        query=query.view(batch_size, 1, head_num * head_size),
        key_cache=kv_cache.key,
        value_cache=kv_cache.value,
        block_list=(
            hpu_attention_meta.block_list
            if window_size_left == -1
            else hpu_attention_meta.block_list_in_window
        ),
        block_mapping=(
            hpu_attention_meta.block_mapping
            if window_size_left == -1
            else hpu_attention_meta.block_mapping_in_window
        ),
        block_bias=(
            hpu_attention_meta.attn_bias
            if window_size_left == -1
            else hpu_attention_meta.attn_bias_in_window
        ),
        block_groups=(
            hpu_attention_meta.block_groups
            if window_size_left == -1
            else hpu_attention_meta.block_groups_in_window
        ),
        block_size=BLOCK_SIZE,
        scale=softmax_scale,
        matmul_qk_op=FP8Matmul(kv_scales.key_scale) if fp8_kv else Matmul(),
        matmul_av_op=FP8Matmul(kv_scales.value_scale) if fp8_kv else Matmul(),
        batch2block_matmul_op=Matmul(),
        block2batch_matmul_op=Matmul(),
        keys_fetch_func=FetchFromCache(1.0 / kv_scales.key_scale_cpu),
        values_fetch_func=FetchFromCache(1.0 / kv_scales.value_scale_cpu),
    )
    # Reshape the output tensor.
    return output.view(batch_size, head_num, head_size)


def paged_attention_mla(
    query: torch.Tensor,
    kv_cache: KVCache,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    seqlen: Seqlen,
    *,
    kv_scales: KVScales,
    softcap: Optional[float] = None,
    hpu_attention_meta: HPUPagedAttentionMetadata,
    kv_lora_rank: int = 0,
):
    batch_size, head_num, head_size = query.shape
    fp8_kv = kv_cache.dtype == torch.float8_e4m3fn
    output = ops.flat_pa_mla(
        query=query,
        key_cache=kv_cache.key,
        value_cache=None,
        block_list=hpu_attention_meta.block_list,
        block_mapping=hpu_attention_meta.block_mapping,
        block_bias=hpu_attention_meta.attn_bias,
        block_groups=hpu_attention_meta.block_groups,
        block_size=BLOCK_SIZE,
        scale=softmax_scale,
        matmul_qk_op=FP8Matmul(kv_scales.key_scale) if fp8_kv else Matmul(),
        matmul_av_op=FP8Matmul(kv_scales.value_scale) if fp8_kv else Matmul(),
        batch2block_matmul_op=Matmul(),
        block2batch_matmul_op=Matmul(),
        keys_fetch_func=FetchFromCache(1.0 / kv_scales.key_scale_cpu),
        values_fetch_func=None,
        kv_lora_rank=kv_lora_rank,
    )
    # Reshape the output tensor.
    return output.view(batch_size, head_num, -1)


__all__ = [
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
    "paged_attention_mla",
    "set_block_mapping",
]
