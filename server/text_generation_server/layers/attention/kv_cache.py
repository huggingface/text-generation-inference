from typing import Tuple

import torch
from text_generation_server.models.globals import ATTENTION, BLOCK_SIZE
from text_generation_server.utils.import_utils import SYSTEM


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

        if dtype in {torch.float8_e5m2, torch.float8_e4m3fn} and (
            ATTENTION != "flashinfer" or SYSTEM != "cuda"
        ):
            raise ValueError(
                "FP8 KV cache is currently only supported for flashinfer on CUDA"
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
    ):
        """Store the key and value at the given slots."""

        key_cache = self.kv_cache[0]
        value_cache = self.kv_cache[1]

        if ATTENTION in {"flashdecoding", "flashinfer"}:
            # TODO: add scale
            key = key.to(key_cache.dtype)
            value = value.to(value_cache.dtype)
            if key_cache.dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
                # Torch index_put does not support float8_{e5m2,e4m3fn} yet, so
                # put as raw data instead.
                key_cache = key_cache.view(torch.uint8)
                value_cache = value_cache.view(torch.uint8)
                key = key.view(torch.uint8)
                value = value.view(torch.uint8)
            shape = key_cache.shape
            key_cache.view(-1, shape[-2], shape[-1])[slots] = key
            value_cache.view(-1, shape[-2], shape[-1])[slots] = value
        else:
            paged_reshape_and_cache(key, value, key_cache, value_cache, slots)


def paged_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
):
    if SYSTEM == "cuda":
        try:
            from vllm._C import cache_ops
        except Exception as e:
            raise ImportError(
                f"Could not import vllm paged attention. Make sure your installation is correct. Complete error: {e}"
            )
        cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slots, "auto", 1.0
        )
    elif SYSTEM == "rocm":
        try:
            import vllm._custom_ops as ops
        except Exception as e:
            raise ImportError(
                f"Could not import vllm paged attention. Make sure your installation is correct. Complete error: {e}"
            )
        ops.reshape_and_cache(key, value, key_cache, value_cache, slots, "auto", 1.0)
    elif SYSTEM == "ipex":
        import intel_extension_for_pytorch as ipex

        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache, slots
        )
    else:
        raise NotImplementedError(
            f"Cannot reshape and cache for paged attention, system '{SYSTEM}' not supportedattention"
        )
