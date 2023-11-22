import torch
import my_custom_comm
from mpi4py import MPI

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
world_size = COMM.Get_size()
tp_ptr, pp_ptr = my_custom_comm.init_nccl(2, 1)
print(tp_ptr)
device = rank % torch.cuda.device_count()
torch.cuda.set_device(device)
torch.cuda.set_per_process_memory_fraction(1., device)
print(rank, world_size)

t = torch.tensor([[1, 2, 3, 4], [3, 3, 3, 3.1]],
                 dtype=torch.float16).to('cuda')

print(my_custom_comm.custom_allreduce(t, tp_ptr))
print(t)

torch.cuda.synchronize()
my_custom_comm.finalize_nccl(tp_ptr, pp_ptr)