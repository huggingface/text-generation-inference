import torch
from loguru import logger


def get_hpu_free_memory(device, memory_fraction):
    from habana_frameworks.torch.hpu import memory_stats

    device_id = device.index
    mem_stats = memory_stats(device_id)
    logger.info(f"mem_stats: {mem_stats}")
    total_free_memory = mem_stats["Limit"] - mem_stats["MaxInUse"]
    free_memory = max(
        0, int(total_free_memory - (1 - memory_fraction) * mem_stats["Limit"])
    )
    return free_memory


def synchronize_hpu(device):
    torch.hpu.synchronize()


def noop(*args, **kwargs):
    pass


empty_cache = noop
synchronize = synchronize_hpu
get_free_memory = get_hpu_free_memory
