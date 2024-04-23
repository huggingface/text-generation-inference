import os
import torch

from loguru import logger

from text_generation_server.utils.import_utils import IS_CUDA_SYSTEM, IS_ROCM_SYSTEM
from text_generation_server.utils.flash_attn_triton import triton_attention

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")

if not torch.cuda.is_available():
    raise ImportError("CUDA is not available")

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5
is_sm8x = major == 8 and minor >= 0
is_sm90 = major == 9 and minor == 0
is_sm94 = major == 9 and minor == 4

HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2_CUDA = False
HAS_FLASH_ATTN_V2_ROCM = False

ROCM_USE_FLASH_ATTN_V2_CK = False
ROCM_USE_FLASH_ATTN_V2_TRITON = False

if IS_ROCM_SYSTEM:
    if os.getenv("ROCM_USE_FLASH_ATTN_V2_TRITON", "").lower() == "true":
        ROCM_USE_FLASH_ATTN_V2_TRITON = True
        logger.info("ROCm: using Flash Attention 2 Triton implementation.")
    else:
        ROCM_USE_FLASH_ATTN_V2_CK = True
        logger.info("ROCm: using Flash Attention 2 Composable Kernel implementation.")

try:
    try:
        import flash_attn_2_cuda
    except ImportError:
        architecture_suffix = ""
        if IS_CUDA_SYSTEM:
            architecture_suffix = "-cuda"
        elif IS_ROCM_SYSTEM:
            architecture_suffix = "-rocm"
        raise ImportError(
            "Flash Attention V2 is not installed.\n"
            "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
            f"or install flash attention v2 with `cd server && make install install-flash-attention-v2{architecture_suffix}`"
        )
    if IS_CUDA_SYSTEM and not (is_sm8x or is_sm90):
        raise ImportError(
            f"GPU with CUDA capability {major} {minor} is not supported for "
            "Flash Attention V2"
        )
    elif IS_ROCM_SYSTEM and not (is_sm8x or is_sm90 or is_sm94):
        raise ImportError(
            f"AMD GPU with compute capability {major} {minor} is not supported for "
            "Flash Attention V2"
        )
    HAS_FLASH_ATTN_V2_CUDA = IS_CUDA_SYSTEM
    HAS_FLASH_ATTN_V2_ROCM = IS_ROCM_SYSTEM
except ImportError as e:
    try:
        import flash_attn_cuda
    except ImportError:
        raise ImportError(
            "Flash Attention is not installed.\n"
            "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
            "or install flash attention with `cd server && make install install-flash-attention`"
        ) from e

    if IS_CUDA_SYSTEM and not (is_sm75 or is_sm8x or is_sm90):
        raise ImportError(
            f"GPU with CUDA capability {major} {minor} is not supported"
        ) from e
    elif IS_ROCM_SYSTEM:
        for idx in range(torch.cuda.device_count()):
            if "MI210" not in torch.cuda.get_device_name(
                idx
            ) and "MI250" not in torch.cuda.get_device_name(idx):
                raise ImportError(
                    f"AMD GPU {torch.cuda.get_device_name(idx)} does not support flash-attention"
                )

    logger.warning(f"Unable to use Flash Attention V2: {e}")
    HAS_FLASH_ATTN = True


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    total_tokens, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(
        total_tokens, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(total_tokens, num_key_value_heads * n_rep, head_dim)


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
    if window_size_left <= 0 and window_size_left != -1:
        raise ValueError("`window_size_left` must be > 0 or -1")

    if IS_CUDA_SYSTEM and HAS_FLASH_ATTN_V2_CUDA:
        return flash_attn_2_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            None,
            max_s,
            max_s,
            0.0,
            softmax_scale,
            False,
            True,
            window_size_left,
            0,
            False,
            None,
        )
    elif IS_CUDA_SYSTEM and HAS_FLASH_ATTN:
        if window_size_left != -1:
            raise NotImplementedError(
                "window_size_left is only available with flash attn v2"
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

        return flash_attn_cuda.fwd(
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
            0,
            None,
        )
    elif IS_ROCM_SYSTEM and HAS_FLASH_ATTN_V2_ROCM and ROCM_USE_FLASH_ATTN_V2_CK:
        if window_size_left != -1:
            raise ValueError(
                f"RoCm version of Flash Attention v2 does not support window attention (window_size_left != -1, got window_size_left={window_size_left})."
            )

        # RoCm flash API does not take the window_size_left and window_size_right arguments.
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
            True,
            False,
            None,
        )
    elif IS_ROCM_SYSTEM and ROCM_USE_FLASH_ATTN_V2_TRITON:
        # NOTE: The Triton kernel silently outputs wrong results when using MQA/GQA and not
        # repeating.
        # TODO: just a sketch. Kind of need to abstract this `attention` function to enable some customization and pass those - let's sync with Nicolas for which implem he'd like
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_kv_heads != num_heads:
            # Interleave for MQA workaround.
            k = repeat_kv(k, num_heads // num_kv_heads)
            v = repeat_kv(v, num_heads // num_kv_heads)

        output, _ = triton_attention(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            True,
            softmax_scale,
        )
        return output
    else:
        raise NotImplementedError(
            f"Flash attention is not installed (IS_CUDA_SYSTEM={IS_CUDA_SYSTEM}, IS_ROCM_SYSTEM={IS_ROCM_SYSTEM}, HAS_FLASH_ATTN_V2_CUDA={HAS_FLASH_ATTN_V2_CUDA}, HAS_FLASH_ATTN_V2_ROCM={HAS_FLASH_ATTN_V2_ROCM})"
        )
