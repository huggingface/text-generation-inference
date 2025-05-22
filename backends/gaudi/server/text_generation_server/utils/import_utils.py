import torch


def get_hpu_free_memory(device, memory_fraction):
    free_hpu_memory, _ = torch.hpu.mem_get_info()
    return free_hpu_memory


def synchronize_hpu(device):
    torch.hpu.synchronize()


def noop(*args, **kwargs):
    pass


empty_cache = noop
synchronize = synchronize_hpu
get_free_memory = get_hpu_free_memory
