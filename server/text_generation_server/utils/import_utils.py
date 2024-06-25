import torch
from loguru import logger
import subprocess


def is_ipex_available():
    try:
        import intel_extension_for_pytorch
    except ImportError:
        return False
    return True


def get_cuda_free_memory(device, memory_fraction):
    total_free_memory, _ = torch.cuda.mem_get_info(device)
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory = max(0, total_free_memory - (1 - memory_fraction) * total_gpu_memory)
    return free_memory


def get_xpu_free_memory(device, memory_fraction):
    total_memory = torch.xpu.get_device_properties(device).total_memory
    device_id = device.index
    query = f"xpu-smi dump -d {device_id} -m 18 -n 1"
    output = subprocess.check_output(query.split()).decode("utf-8").split("\n")
    used_memory = float(output[1].split(",")[-1]) * 1024 * 1024
    free_memory = int(total_memory * 0.95 - used_memory)
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
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        empty_cache = torch.xpu.empty_cache
        synchronize = torch.xpu.synchronize
        get_free_memory = get_xpu_free_memory
    else:
        empty_cache = noop
        synchronize = noop
        get_free_memory = get_cpu_free_memory
else:
    SYSTEM = "cpu"

    empty_cache = noop
    synchronize = noop
    get_free_memory = get_cpu_free_memory
logger.info(f"Detected system {SYSTEM}")
