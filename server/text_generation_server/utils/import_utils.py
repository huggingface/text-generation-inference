import torch
from loguru import logger
import os


import importlib.util


def is_ipex_available():
    return importlib.util.find_spec("intel_extension_for_pytorch") is not None


def get_cuda_free_memory(device, memory_fraction):
    total_free_memory, _ = torch.cuda.mem_get_info(device)
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory = max(0, total_free_memory - (1 - memory_fraction) * total_gpu_memory)
    return free_memory


def get_xpu_free_memory(device, memory_fraction):
    total_free_memory, total_xpu_memory = torch.xpu.mem_get_info(device)
    memory_fraction = float(os.getenv("XPU_MEMORY_FRACTION", "0.9"))
    free_memory = max(
        0, int(total_free_memory - (1 - memory_fraction) * total_xpu_memory)
    )
    return free_memory


def get_cpu_free_memory(device, memory_fraction):
    import psutil
    from text_generation_server.utils.dist import WORLD_SIZE

    mem = psutil.virtual_memory()
    free_memory = int(mem.available * 0.95 / WORLD_SIZE)
    return free_memory


def noop(*args, **kwargs):
    pass


SYSTEM = None
if torch.version.hip is not None:
    SYSTEM = "rocm"
    empty_cache = torch.cuda.empty_cache
    synchronize = torch.cuda.synchronize
    get_free_memory = get_cuda_free_memory
elif torch.version.cuda is not None and torch.cuda.is_available():
    SYSTEM = "cuda"
    empty_cache = torch.cuda.empty_cache
    synchronize = torch.cuda.synchronize
    get_free_memory = get_cuda_free_memory
elif is_ipex_available():
    SYSTEM = "ipex"
    import intel_extension_for_pytorch  # noqa: F401

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        empty_cache = torch.xpu.empty_cache
        synchronize = torch.xpu.synchronize
        get_free_memory = get_xpu_free_memory
    else:
        empty_cache = noop
        synchronize = noop
        get_free_memory = get_cpu_free_memory
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    SYSTEM = "xpu"
    empty_cache = torch.xpu.empty_cache
    synchronize = torch.xpu.synchronize
    get_free_memory = get_xpu_free_memory
else:
    SYSTEM = "cpu"

    empty_cache = noop
    synchronize = noop
    get_free_memory = get_cpu_free_memory
logger.info(f"Detected system {SYSTEM}")
