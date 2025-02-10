from typing import Tuple
from dataclasses import dataclass, field

from loguru import logger
import torch

from text_generation_server.layers.fp8 import fp8_quantize
from text_generation_server.models.globals import ATTENTION, BLOCK_SIZE
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.kernels import load_kernel
from text_generation_server.utils.log import log_once
from text_generation_server.utils.weights import Weights

if SYSTEM == "cuda":
    try:
        paged_attention = load_kernel(
            module="paged_attention", repo_id="kernels-community/paged-attention"
        )
    except Exception as e:
        raise ImportError(
            f"Could not import attention kernels. Make sure your installation is correct. Complete error: {e}"
        )
else:
    paged_attention = None


@dataclass
class KVScales:
    """
    Key-value scales for FP8 KV cache.

    This data class stores key and value scales both as a GPU tensor and
    as a GPU float. This inconvenience is necessary because some functions
    (e.g. scaling kernels) take scales as a GPU tensor, whereas others
    (e.g. flashinfer) take scales as a CPU scalar.
    """

    key_scale: torch.Tensor
    value_scale: torch.Tensor
    key_scale_cpu: float = field(init=False)
    value_scale_cpu: float = field(init=False)

    def __post_init__(self):
        if self.key_scale.numel() != 1 or self.value_scale.numel() != 1:
            raise ValueError("Key and value scales must be scalar tensors.")

        self.key_scale_cpu = self.key_scale.item()
        self.value_scale_cpu = self.value_scale.item()


class KVCache:
    """
    Key-value cache for attention layers.
    """

    kv_cache: Tuple[torch.Tensor, torch.Tensor]

    def __init__(
        self,
        *,
        num_blocks: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Construct the key-value cache for a layer."""
        if dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
            if not (
                (ATTENTION == "flashinfer" and SYSTEM == "cuda")
                or (ATTENTION == "paged" and SYSTEM in ("cuda", "rocm"))
            ):
                raise ValueError(
                    "FP8 KV cache is currently only supported for flashinfer on CUDA and paged attention on CUDA and ROCm. "
                )
            if SYSTEM == "rocm" and dtype == torch.float8_e5m2:
                raise ValueError(
                    "float8_e5m2 FP8 KV cache is not supported on AMD ROCm"
                )

        element_size = torch.tensor([], dtype=dtype).element_size()
        if SYSTEM == "ipex" and device.type == "xpu":
            x = 1
        else:
            x = BLOCK_SIZE // element_size

        if ATTENTION in {"flashdecoding", "flashinfer"} or (
            ATTENTION == "flashdecoding-ipex" and device.type == "xpu"
        ):
            self.kv_cache = (
                torch.empty(
                    (num_blocks, BLOCK_SIZE, num_heads, head_size),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_blocks, BLOCK_SIZE, num_heads, head_size),
                    dtype=dtype,
                    device=device,
                ),
            )
        elif SYSTEM == "ipex" and device == torch.device("cpu"):
            # ipex cpu flashdecoding kernel and paged attention kernel share same layout
            self.kv_cache = (
                torch.empty(
                    (num_blocks, num_heads, BLOCK_SIZE, head_size),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_blocks, num_heads, BLOCK_SIZE, head_size),
                    dtype=dtype,
                    device=device,
                ),
            )
        else:
            self.kv_cache = (
                torch.zeros(
                    (num_blocks, num_heads, head_size // x, BLOCK_SIZE, x),
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    (num_blocks, num_heads, head_size, BLOCK_SIZE),
                    dtype=dtype,
                    device=device,
                ),
            )

    def can_scale(self, kv_scales: KVScales) -> bool:
        """Check if the cache can be scaled by the given scales."""
        if kv_scales.key_scale_cpu == 1.0 and kv_scales.value_scale_cpu == 1.0:
            return False
        elif self.dtype == torch.float8_e4m3fn and (
            (ATTENTION in ("paged", "flashinfer") and SYSTEM == "cuda")
            or (ATTENTION == "paged" and SYSTEM == "rocm")
        ):
            log_once(logger.info, "Using FP8 KV cache scales")
            return True
        else:
            # We have scales, but not the correct FP8 cache type, so warn once.
            log_once(
                logger.info,
                "Ignoring FP8 KV cache scales, supported only for float8_e4m3fn KV cache with flashinfer on CUDA and paged attention on ROCm",
            )
            return False

    @property
    def dtype(self):
        """Get the data type of the cache."""
        return self.kv_cache[0].dtype

    @property
    def key(self):
        """Get the key cache."""

        return self.kv_cache[0]

    @property
    def value(self):
        """Get the value cache."""

        return self.kv_cache[1]

    def store(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        kv_scales: KVScales,
    ):
        """Store the key and value at the given slots."""

        key_cache = self.kv_cache[0]
        value_cache = self.kv_cache[1]

        if self.can_scale(kv_scales) and SYSTEM == "cuda":
            if kv_scales.key_scale_cpu != 1.0:
                key = fp8_quantize(
                    key.float(),
                    scale=kv_scales.key_scale,
                    qdtype=self.dtype,
                    scalar=True,
                )[0]
            if kv_scales.value_scale_cpu != 1.0:
                value = fp8_quantize(
                    value.float(),
                    scale=kv_scales.value_scale,
                    qdtype=self.dtype,
                    scalar=True,
                )[0]

        if ATTENTION in {"flashdecoding", "flashinfer"}:
            key = key.to(key_cache.dtype)
            value = value.to(value_cache.dtype)
            if key_cache.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
                # Torch index_put does not support float8_{e5m2,e4m3fn} yet, so
                # put as raw data instead.
                key_cache = key_cache.view(torch.uint8)
                value_cache = value_cache.view(torch.uint8)
                key = key.view(torch.uint8)
                value = value.view(torch.uint8)
            shape = key_cache.shape
            key_cache.view(-1, shape[-2], shape[-1])[slots] = key
            value_cache.view(-1, shape[-2], shape[-1])[slots] = value
        elif ATTENTION == "flashdecoding-ipex" and key.device.type == "xpu":
            import intel_extension_for_pytorch as ipex

            ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
                key, value, key_cache, value_cache, slots
            )
        else:
            paged_reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slots,
                kv_scales.key_scale_cpu,
                kv_scales.value_scale_cpu,
            )


def paged_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):

    if SYSTEM == "cuda":
        kv_cache_dtype = "auto"
        if key_cache.dtype == torch.float8_e4m3fn:
            kv_cache_dtype = "fp8"

        paged_attention.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slots,
            kv_cache_dtype,
            torch.tensor(k_scale),
            torch.tensor(v_scale),
        )
    elif SYSTEM == "rocm":
        try:
            import vllm._custom_ops as ops
        except Exception as e:
            raise ImportError(
                f"Could not import vllm paged attention. Make sure your installation is correct. Complete error: {e}"
            )

        kv_cache_dtype = "auto"
        if key_cache.dtype == torch.float8_e4m3fn:
            key_cache = key_cache.view(torch.uint8)
            value_cache = value_cache.view(torch.uint8)
            kv_cache_dtype = "fp8"

        ops.reshape_and_cache(
            key, value, key_cache, value_cache, slots, kv_cache_dtype, k_scale, v_scale
        )
    elif SYSTEM == "ipex":
        import intel_extension_for_pytorch as ipex

        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache, slots
        )
    else:
        raise NotImplementedError(
            f"Cannot reshape and cache for paged attention, system '{SYSTEM}' not supported"
        )


def get_kv_scales(weights: Weights, prefix: str) -> KVScales:
    """Load KV cache scales."""

    key_scale = torch.tensor(1.0, dtype=torch.float32, device=weights.device)
    value_scale = key_scale
    if weights.has_tensor(f"{prefix}.k_scale") and weights.has_tensor(
        f"{prefix}.v_scale"
    ):
        key_scale = weights.get_tensor(f"{prefix}.k_scale", to_dtype=False).float()
        value_scale = weights.get_tensor(f"{prefix}.v_scale", to_dtype=False).float()
    elif weights.has_tensor(f"{prefix}.kv_scale"):
        # Fall back to older more coarse-grained scale when available.
        key_scale = weights.get_tensor(f"{prefix}.kv_scale").float()
        value_scale = key_scale

    return KVScales(key_scale=key_scale, value_scale=value_scale)
