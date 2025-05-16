import torch
from loguru import logger
import habana_frameworks.torch as htorch
import os


def get_hpu_free_memory(device, memory_fraction):
    graph_reserved_mem = (
        float(os.environ.get("TGI_GRAPH_RESERVED_MEM", "0.1"))
        if htorch.utils.internal.is_lazy()
        else 0
    )
    free_memory = int(
        torch.hpu.mem_get_info()[0] * memory_fraction * (1 - graph_reserved_mem)
    )
    logger.info(f"Free memory on device {device}: {free_memory} bytes.")
    return free_memory


def synchronize_hpu(device):
    torch.hpu.synchronize()


def noop(*args, **kwargs):
    pass


empty_cache = noop
synchronize = synchronize_hpu
get_free_memory = get_hpu_free_memory
