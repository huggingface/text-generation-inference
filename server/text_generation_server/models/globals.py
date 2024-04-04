import torch
import os

MEM_POOL = torch.cuda.graph_pool_handle()
# This is overridden by the cli
cuda_graphs = os.getenv("CUDA_GRAPHS")
if cuda_graphs is not None:
    try:
        cuda_graphs = [int(item) for item in cuda_graphs.split(",")]
    except Exception as e:
        raise RuntimeError(
            f"Could not parse cuda graphs {cuda_graphs}, expected comma separated list for batch sizes to run on: {e}"
        )
CUDA_GRAPHS = cuda_graphs
