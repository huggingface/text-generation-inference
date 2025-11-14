import os
import torch
from torch.distributed import ProcessGroup
from datetime import timedelta
from loguru import logger

# Tensor Parallelism settings
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
MEMORY_FRACTION = float(os.getenv("HPU_MEMORY_FRACTION", "0.8"))


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup(ProcessGroup):
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size
        super().__init__(rank, size)

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

    def _get_backend_name(self):
        return "fake"


def initialize_torch_distributed():
    if WORLD_SIZE == 1:
        return FakeGroup(RANK, WORLD_SIZE), RANK, WORLD_SIZE
    else:
        if os.getenv("DEBUG", None) == "1":
            return FakeGroup(RANK, WORLD_SIZE), RANK, WORLD_SIZE

        if not torch.distributed.is_initialized():
            # Call the init process.
            torch.distributed.init_process_group(
                backend="hccl",
                world_size=WORLD_SIZE,
                rank=RANK,
                timeout=timedelta(seconds=120),
            )
        else:
            logger.warning("torch.distributed is already initialized.")

        return torch.distributed.group.WORLD, RANK, WORLD_SIZE
