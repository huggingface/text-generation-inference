import torch
import os

MEM_POOL = torch.cuda.graph_pool_handle()
# This is overridden by the cli
ENABLE_CUDA_GRAPHS = os.getenv("ENABLE_CUDA_GRAPHS", "false").lower() in {"1", "true"}
