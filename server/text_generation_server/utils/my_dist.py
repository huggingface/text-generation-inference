import os
import torch

from datetime import timedelta
from loguru import logger
from mpi4py import MPI
import my_custom_comm

# Tensor Parallelism settings
RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))

# CUDA memory fraction
MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))

class MyCommGroup:
    def __init__(self, rank, size, tp_comm, pp_comm):
        self._rank = rank
        self._size = size
        self.tp_comm = tp_comm
        self.pp_comm = pp_comm

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def initialize_mpi_distributed():
    assert torch.cuda.is_available()
    # mpi initialize
    COMM = MPI.COMM_WORLD
    assert COMM.Get_size() == WORLD_SIZE, f"{COMM.Get_size()},{WORLD_SIZE}"
    
    # Set the device id.
    assert WORLD_SIZE <= torch.cuda.device_count(), "Each process is one gpu"
    device = RANK % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    
    # nccl initialize
    tp_comm, pp_comm = my_custom_comm.init_nccl(WORLD_SIZE, 1)
    process_group = MyCommGroup(RANK, WORLD_SIZE, tp_comm, pp_comm)
    
    logger.warning("custom mpi and nccl is already initialized.")
    return process_group, RANK, WORLD_SIZE, COMM
