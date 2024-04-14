import torch
import os

MEM_POOL = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
# This is overridden by the cli
ENABLE_CUDA_GRAPHS = os.getenv("ENABLE_CUDA_GRAPHS", "false").lower() in {"1", "true"}
