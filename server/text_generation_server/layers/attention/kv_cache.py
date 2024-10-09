from typing import Tuple

from loguru import logger
import torch

from text_generation_server.models.globals import ATTENTION, BLOCK_SIZE
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.log import log_once
from text_generation_server.utils.weights import Weights


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

        if dtype == torch.float8_e5m2 and (
            ATTENTION != "flashinfer" or SYSTEM != "cuda"
        ):
            raise ValueError(
                "float8_e5m2 KV cache is currently only supported for flashinfer on CUDA"
            )

        element_size = torch.tensor([], dtype=dtype).element_size()
        if SYSTEM == "ipex" and device.type == "xpu":
            x = 1
        else:
            x = BLOCK_SIZE // element_size

        if ATTENTION in {"flashdecoding", "flashinfer"}:
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
        key_scale: float,
        value_scale: float,
    ):
        """Store the key and value at the given slots."""

        key_cache = self.kv_cache[0]
        value_cache = self.kv_cache[1]

        # Only use scales when we have a float8 KV cache.
        if self.dtype == torch.float8_e5m2:
            if key_scale != 1.0:
                key = key * key_scale
            if value_scale != 1.0:
                value = value * value_scale
        elif key_scale != 1.0 or value_scale != 1.0:
            log_once(
                logger.info,
                f"Ignoring FP8 KV cache scales, cache type: {self.dtype}",
            )

        if ATTENTION in {"flashdecoding", "flashinfer"}:
            key = key.to(key_cache.dtype)
            value = value.to(value_cache.dtype)
            if self.dtype == torch.float8_e5m2:
                # Torch index_put does not support float8_e5m2 yet, so
                # put as raw data instead.
                key_cache = key_cache.view(torch.uint8)
                value_cache = value_cache.view(torch.uint8)
                key = key.view(torch.uint8)
                value = value.view(torch.uint8)
            shape = key_cache.shape
            key_cache.view(-1, shape[-2], shape[-1])[slots] = key
            value_cache.view(-1, shape[-2], shape[-1])[slots] = value
        else:
            from text_generation_server.layers.attention import reshape_and_cache

            reshape_and_cache(key, value, key_cache, value_cache, slots)


def get_kv_scales(weights: Weights, prefix: str) -> Tuple[float, float]:
    key_scale = 1.0
    value_scale = 1.0
    if weights._has_tensor(f"{prefix}.k_scale"):
        key_scale = weights.get_scalar(f"{prefix}.k_scale")
    if weights._has_tensor(f"{prefix}.v_scale"):
        value_scale = weights.get_scalar(f"{prefix}.v_scale")

    # Fall back to older more course-grained scale when available.
    if (
        weights._has_tensor(f"{prefix}.kv_scale")
        and key_scale == 1.0
        and value_scale == 1.0
    ):
        key_scale = value_scale = weights.get_scalar(f"{prefix}.kv_scale")
        value_scale = key_scale

    logger.info(f"Using k_scale={key_scale}, v_scale={value_scale} for prefix={prefix}")
    return key_scale, value_scale
