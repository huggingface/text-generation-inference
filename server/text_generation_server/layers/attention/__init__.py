import os

from text_generation_server.utils.import_utils import SYSTEM

from .common import Seqlen

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")
if SYSTEM == "cuda":
    from .cuda import (
        PREFILL_IN_KV_CACHE,
        SUPPORTS_WINDOWING,
        attention,
        paged_attention,
        reshape_and_cache,
    )
elif SYSTEM == "rocm":
    from .rocm import (
        PREFILL_IN_KV_CACHE,
        SUPPORTS_WINDOWING,
        attention,
        paged_attention,
        reshape_and_cache,
    )
elif SYSTEM == "ipex":
    from .ipex import (
        PREFILL_IN_KV_CACHE,
        SUPPORTS_WINDOWING,
        attention,
        paged_attention,
        reshape_and_cache,
    )
else:
    raise ImportError(f"System {SYSTEM} doesn't support flash/paged attention")

# KVCache needs `reshape_and_cache`, so ensure that it is defined already.
from .kv_cache import KVCache

__all__ = [
    "attention",
    "paged_attention",
    "reshape_and_cache",
    "PREFILL_IN_KV_CACHE",
    "SUPPORTS_WINDOWING",
    "KVCache",
    "Seqlen",
]
