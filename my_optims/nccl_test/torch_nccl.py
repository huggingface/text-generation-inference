import torch
import torch.distributed as dist
import os
import time


def test_nccl():
    rank, world_size = int(os.getenv('RANK')), int(os.getenv('WORLD_SIZE'))
    dist.init_process_group(backend='nccl', 
                            rank=rank, 
                            world_size=world_size)
    process_group = dist.group.WORLD
    torch.cuda.set_device(rank % world_size)
    shape = 4096 * 2048
    weight = torch.randn([shape], dtype=torch.float16).to("cuda")
    
    # nccl test
    dist.barrier(process_group)
    for i in range(32):
        torch.cuda.synchronize()
        dist.all_reduce(weight, group=process_group)
        torch.cuda.synchronize()
        weight = torch.randn([shape], dtype=torch.float16).to("cuda")

    dist.barrier(process_group)
    
    
    
if __name__ == '__main__':
    test_nccl()