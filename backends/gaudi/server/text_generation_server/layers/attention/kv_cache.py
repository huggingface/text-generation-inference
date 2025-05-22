from typing import Tuple
from dataclasses import dataclass, field

import torch

from text_generation_server.models.globals import BLOCK_SIZE
from text_generation_server.utils.weights import Weights
from vllm_hpu_extension import cache_ops


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
        ## TODO FP8 kv cache support
        if dtype is torch.float8_e5m2:
            raise ValueError("torch.float8_e5m2 is not supported in hpu. ")

        self.kv_cache = (
            torch.zeros(
                (num_blocks, BLOCK_SIZE, num_heads, head_size),
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                (num_blocks, BLOCK_SIZE, num_heads, head_size),
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
        kv_scales: KVScales,
    ):
        """Store the key and value at the given slots."""
        ## TODO FP8 kv cache support

        key_cache = self.kv_cache[0]
        value_cache = self.kv_cache[1]

        paged_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slots,
            kv_scales.key_scale,
            kv_scales.value_scale,
        )


class KVCompressCache(KVCache):
    """
    Key-value cache for attention layers.
    """

    kv_cache: torch.Tensor

    def __init__(
        self,
        *,
        num_blocks: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Construct the key-value cache for a layer."""
        ## TODO FP8 kv cache support
        if dtype is torch.float8_e5m2:
            raise ValueError("torch.float8_e5m2 is not supported in hpu. ")

        self.kv_cache = torch.zeros(
            (num_blocks, BLOCK_SIZE, 1, head_size),
            dtype=dtype,
            device=device,
        )

    @property
    def dtype(self):
        """Get the data type of the cache."""
        return self.kv_cache.dtype

    @property
    def key(self):
        """Get the key cache."""

        return self.kv_cache

    @property
    def value(self):
        """Get the value cache."""

        return self.kv_cache

    def store(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        kv_scales: KVScales,
    ):
        """Store the key and value at the given slots."""
        ## TODO FP8 kv cache support

        block_idx = slots // BLOCK_SIZE
        block_offset = slots % BLOCK_SIZE
        if self.kv_cache.dtype == torch.float8_e4m3fn:
            key = torch.ops.hpu.cast_to_fp8_v2(
                key, kv_scales.key_scale, False, False, torch.float8_e4m3fn
            )[0]
        cache_ops.insert_or_update_cache(key, self.kv_cache, block_idx, block_offset)


def paged_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
):
    block_idx = slots // BLOCK_SIZE
    block_offset = slots % BLOCK_SIZE
    if key_cache.dtype == torch.float8_e4m3fn:
        key = torch.ops.hpu.cast_to_fp8_v2(
            key, k_scale, False, False, torch.float8_e4m3fn
        )[0]
        value = torch.ops.hpu.cast_to_fp8_v2(
            value, v_scale, False, False, torch.float8_e4m3fn
        )[0]
    cache_ops.insert_or_update_cache(key, key_cache, block_idx, block_offset)
    cache_ops.insert_or_update_cache(value, value_cache, block_idx, block_offset)


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
