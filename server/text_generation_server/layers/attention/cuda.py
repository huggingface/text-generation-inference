import torch
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models.globals import (
    ATTENTION,
    BLOCK_SIZE,
)
from text_generation_server.layers.attention import Seqlen
from typing import Optional

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5
_PARTITION_SIZE = 512

try:
    from vllm._C import cache_ops
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
    if ATTENTION in {"flashdecoding", "flashinfer"}:
        shape = key_cache.shape
        key_cache.view(-1, shape[-2], shape[-1])[slots] = key
        value_cache.view(-1, shape[-2], shape[-1])[slots] = value
    else:
        cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slots, "auto", 1.0
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
    # block_size = value_cache.shape[3]
    block_size = BLOCK_SIZE
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (max_s + _PARTITION_SIZE - 1) // _PARTITION_SIZE

    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    if ATTENTION == "flashinfer":
        from text_generation_server.layers.attention.flashinfer import decode_state

        return decode_state.get().forward(
            query.contiguous(),
            paged_kv_cache=(key_cache, value_cache),
            logits_soft_cap=softcap,
            sm_scale=softmax_scale,
        )
    elif ATTENTION == "flashdecoding":
        max_q = 1
        max_k = max_s
        import flash_attn_2_cuda

        # TODO fixme when flash contains the fix.
        # Number of splits is not correctly handled
        # by the current path
        # https://github.com/Dao-AILab/flash-attention/blob/320fb59487658f033f56711efd3d61b7c7a6f8f3/csrc/flash_attn/flash_api.cpp#L577
        # This fails becuase we're using causal, therefore window_right is set to 0 and the split logic is never applied.
        if softcap is None:
            softcap = 0.0
        out = flash_attn_2_cuda.varlen_fwd(
            query,
            key_cache,
            value_cache,
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
            -1,  # Window_left
            -1,  # Window right
            softcap,
            False,  # return softmax
            None,  # generator
        )
        return out[0]
    else:
        if softcap is not None:
            raise RuntimeError("Paged attention doesn't support softcapping")
        input_lengths = seqlen.input_lengths + seqlen.cache_lengths
        from vllm._C import ops

        out = torch.empty_like(query)

        use_v1 = max_s <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512
        )
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


try:
    is_ampere_or_newer = major >= 8 and minor >= 0
    if not is_ampere_or_newer:
        raise ImportError("FlashAttention only supports Ampere GPUs or newer.")

    import flash_attn_2_cuda

    V2 = True
except ImportError:
    try:
        import flash_attn_cuda

        V2 = False
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
            raise ImportError(
                f"GPU with CUDA capability {major} {minor} is not supported"
            ) from e


SUPPORTS_WINDOWING = V2

if ATTENTION == "flashinfer":

    def attention(
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seqlen: Seqlen,
        block_tables: torch.Tensor,
        softmax_scale,
        window_size_left=-1,
        causal=True,
        softcap=0.0,
    ):
        from text_generation_server.layers.attention.flashinfer import (
            prefill_with_paged_kv_state,
        )

        return prefill_with_paged_kv_state.get().forward(
            q.contiguous(),
            causal=causal,
            paged_kv_cache=(key_cache, value_cache),
            logits_soft_cap=softcap,
            sm_scale=softmax_scale,
            window_left=window_size_left,
        )

elif ATTENTION == "flashdecoding":
    if V2:

        def attention(
            q,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            seqlen: Seqlen,
            block_tables: torch.Tensor,
            softmax_scale,
            window_size_left=-1,
            causal=True,
            softcap=0.0,
        ):
            out = torch.empty_like(q)
            if window_size_left <= 0 and window_size_left != -1:
                raise ValueError("`window_size_left` must be > 0 or -1")
            return flash_attn_2_cuda.varlen_fwd(
                q,
                key_cache,
                value_cache,
                out,
                seqlen.cu_seqlen_q,
                seqlen.cu_seqlen_k,
                None,
                None,
                block_tables,
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

    else:

        def attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seqlen: Seqlen,
            block_tables: torch.Tensor,
            softmax_scale: float,
            window_size_left: int = -1,
            causal: bool = True,
            softcap=None,
        ):
            if window_size_left != -1:
                raise NotImplementedError(
                    "window_size_left is only available with flash attn v2"
                )
            if softcap is not None:
                raise NotImplementedError(
                    "softcap is only available with flash attn v2"
                )

            # Flash attention v1 requires q, k and v to have the same number of heads
            if k.shape[1] != q.shape[1]:
                # MQA expand
                if k.shape[1] == 1:
                    k = k.expand(-1, q.shape[1], -1)
                # Grouped attention reshape
                else:
                    original_shape = k.shape
                    k = (
                        k.unsqueeze(2)
                        .expand(-1, -1, q.shape[1] // k.shape[1], -1)
                        .reshape(original_shape[0], -1, original_shape[2])
                    )
            if v.shape[1] != q.shape[1]:
                # MQA expand
                if v.shape[1] == 1:
                    v = v.expand(-1, q.shape[1], -1)
                # Grouped attention reshape
                else:
                    original_shape = v.shape
                    v = (
                        v.unsqueeze(2)
                        .expand(-1, -1, q.shape[1] // v.shape[1], -1)
                        .reshape(original_shape[0], -1, original_shape[2])
                    )

            out = torch.empty_like(q)
            flash_attn_cuda.fwd(
                q,
                k,
                v,
                out,
                seqlen.cu_seqlen_q,
                seqlen.cu_seqlen_q,
                seqlen.max_q,
                seqlen.max_k,
                0.0,
                softmax_scale,
                False,
                causal,
                False,
                0,
                None,
            )
            return out

elif ATTENTION == "paged":
    if V2:

        def attention(
            q,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            seqlen: Seqlen,
            block_tables: torch.Tensor,
            softmax_scale,
            window_size_left=-1,
            causal=True,
            softcap=0.0,
        ):
            out = torch.empty_like(q)
            if window_size_left <= 0 and window_size_left != -1:
                raise ValueError("`window_size_left` must be > 0 or -1")
            return flash_attn_2_cuda.varlen_fwd(
                q,
                key_cache,
                value_cache,
                out,
                seqlen.cu_seqlen_q,
                seqlen.cu_seqlen_k,
                None,
                None,
                None,  # block_tables,
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

    else:

        def attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seqlen: Seqlen,
            block_tables: torch.Tensor,
            softmax_scale: float,
            window_size_left: int = -1,
            causal: bool = True,
            softcap=None,
        ):
            if window_size_left != -1:
                raise NotImplementedError(
                    "window_size_left is only available with flash attn v2"
                )
            if softcap is not None:
                raise NotImplementedError(
                    "softcap is only available with flash attn v2"
                )

            # Flash attention v1 requires q, k and v to have the same number of heads
            if k.shape[1] != q.shape[1]:
                # MQA expand
                if k.shape[1] == 1:
                    k = k.expand(-1, q.shape[1], -1)
                # Grouped attention reshape
                else:
                    original_shape = k.shape
                    k = (
                        k.unsqueeze(2)
                        .expand(-1, -1, q.shape[1] // k.shape[1], -1)
                        .reshape(original_shape[0], -1, original_shape[2])
                    )
            if v.shape[1] != q.shape[1]:
                # MQA expand
                if v.shape[1] == 1:
                    v = v.expand(-1, q.shape[1], -1)
                # Grouped attention reshape
                else:
                    original_shape = v.shape
                    v = (
                        v.unsqueeze(2)
                        .expand(-1, -1, q.shape[1] // v.shape[1], -1)
                        .reshape(original_shape[0], -1, original_shape[2])
                    )

            out = torch.empty_like(q)
            flash_attn_cuda.fwd(
                q,
                k,
                v,
                out,
                seqlen.cu_seqlen_q,
                seqlen.cu_seqlen_q,
                seqlen.max_q,
                seqlen.max_k,
                0.0,
                softmax_scale,
                False,
                causal,
                False,
                0,
                None,
            )
            return out

else:
    raise RuntimeError(f"Unknwon attention {ATTENTION}")


# Prefill in the cache with every kind of attention, unless we
# have a configuration that requires flash-attention v1, which
# does not support block tables.
PREFILL_IN_KV_CACHE = ATTENTION == "flashinfer" or (ATTENTION == "flashdecoding" and V2)

__all__ = [
    "PREFILL_IN_KV_CACHE",
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
    "reshape_and_cache",
]
