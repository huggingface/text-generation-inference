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

t = torch.tensor([rank+3.1]*4, dtype=torch.float16).to('cuda').view((1,-1))
print(t.shape)
world_out = t.new_empty(1, 8)
print(my_custom_comm.custom_allgather_into_tensor(world_out, t, tp_ptr))
print(my_custom_comm.custom_allreduce(t, tp_ptr))
print(t, world_out)
torch.cuda.synchronize()


my_custom_comm.finalize_nccl(tp_ptr, pp_ptr)