import os
import torch
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models.globals import FLASH_DECODING
from text_generation_server.layers.attention import Seqlen
from loguru import logger

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5
_PARTITION_SIZE = 512

use_triton = os.getenv("ROCM_USE_FLASH_ATTN_V2_TRITON", "").lower() in {"true", "1"}
ENGINE = "triton" if use_triton else "ck"

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
    if FLASH_DECODING:
        shape = key_cache.shape
        key_cache.view(-1, shape[-2], shape[-1])[slots] = key
        value_cache.view(-1, shape[-2], shape[-1])[slots] = value
    else:
        cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slots, "auto", 1.0
        )


def paged_attention(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    input_lengths: Seqlen,
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
    input_lengths = input_lengths.input_lengths

    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    from vllm._C import ops

    use_v1 = max_s <= 8192 and (max_num_partitions == 1 or num_seqs * num_heads > 512)
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
    return out


if ENGINE != "triton":
    try:
        import flash_attn_2_cuda

        logger.info("ROCm: using Flash Attention 2 Composable Kernel implementation.")
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
if ENGINE == "ck":

    def attention(
        q,
        k,
        v,
        out,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
        causal=True,
    ):
        if window_size_left <= 0 and window_size_left != -1:
            raise ValueError("`window_size_left` must be > 0 or -1")

        # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
        return flash_attn_2_cuda.varlen_fwd(
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

elif ENGINE == "triton":
    from .flash_attn_triton import triton_attention

    def attention(
        q,
        k,
        v,
        out,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
        causal=True,
    ):
        # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
        output, _ = triton_attention(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            causal,
            softmax_scale,
        )
        return output

else:
    raise RuntimeError(f"Unknown attention engine {ENGINE}")
