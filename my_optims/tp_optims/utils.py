# encoding:utf-8
# -------------------------------------------#
# Filename: optims -- utils.py
#
# Description:   
# Version:       1.0
# Created:       2023/9/27-14:43
# Last modified by: 
# Author:        'zhaohuayang@myhexin.com'
# Company:       同花顺网络信息股份有限公司
# -------------------------------------------#
import os
import time
from datetime import timedelta

import torch
from loguru import logger
from torch.distributed import ProcessGroupNCCL

RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "2"))
NCCL_PORT = int(os.getenv("NCCL_PORT", "29500"))
MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def allreduce(self, *args, **kwargs):
        return FakeBarrier()

    def allgather(self, inputs, local_tensor, **kwargs):
        assert (
                len(inputs[0]) == len(local_tensor) == 1
        ), f"{len(inputs[0])} != {len(local_tensor)} != 1, and the FakeGroup is supposed to join on simple tensors"
        for input_ in inputs:
            input_[0].data = local_tensor[0].data
        return FakeBarrier()

    def barrier(self, *args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def initialize_torch_distributed():
    # Set the device id.
    assert WORLD_SIZE <= torch.cuda.device_count(), "Each process is one gpu"
    device = RANK % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    backend = "nccl"
    options = ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    options._timeout = timedelta(seconds=60)
    if not torch.distributed.is_initialized():
        # Call the init process.
        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://localhost:{NCCL_PORT}",
            world_size=WORLD_SIZE,
            rank=RANK,
            timeout=timedelta(seconds=60),
            pg_options=options,
        )
        logger.info(f"torch.distributed is initialized on rank {RANK} of {WORLD_SIZE}.")
    else:
        logger.warning("torch.distributed is already initialized.")

    return torch.distributed.group.WORLD, RANK, WORLD_SIZE


class Timer:
    def __init__(self):
        self.times = []
        self.time = None

    def start(self):
        torch.cuda.synchronize()
        self.time = time.time()

    def end(self):
        torch.cuda.synchronize()
        self.times.append(time.time() - self.time)

    @property
    def elapsed(self):
        self.times.pop(0)
        return round(sum(self.times) / len(self.times) * 1000, 2)
