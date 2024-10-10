import os

from text_generation_server.utils.import_utils import SYSTEM

from .common import Seqlen

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")
if SYSTEM == "cuda":
    from .cuda import (
        SUPPORTS_WINDOWING,
        attention,
        paged_attention,
    )
elif SYSTEM == "rocm":
    from .rocm import (
        SUPPORTS_WINDOWING,
        attention,
        paged_attention,
    )
elif SYSTEM == "ipex":
    from .ipex import (
        SUPPORTS_WINDOWING,
        attention,
        paged_attention,
    )
else:
    raise ImportError(f"System {SYSTEM} doesn't support flash/paged attention")

# KVCache needs `reshape_and_cache`, so ensure that it is defined already.
from .kv_cache import KVCache

__all__ = [
    "attention",
    "paged_attention",
    "SUPPORTS_WINDOWING",
    "KVCache",
    "Seqlen",
]
