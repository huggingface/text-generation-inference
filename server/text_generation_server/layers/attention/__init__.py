from text_generation_server.utils.import_utils import SYSTEM
import os

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")
if SYSTEM == "cuda":
    from .cuda import attention, paged_attention, reshape_and_cache, SUPPORTS_WINDOWING
elif SYSTEM == "rocm":
    from .rocm import attention, paged_attention, reshape_and_cache, SUPPORTS_WINDOWING
elif SYSTEM == "ipex":
    from .ipex import attention, paged_attention, reshape_and_cache, SUPPORTS_WINDOWING
else:
    raise ImportError(f"System {SYSTEM} doesn't support flash/paged attention")
