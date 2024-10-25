from text_generation_server.utils.import_utils import SYSTEM

if SYSTEM == "ipex":
    from .ipex import WQLinear
elif SYSTEM == "cuda":
    from .cuda import WQLinear

__all__ = ["WQLinear"]
