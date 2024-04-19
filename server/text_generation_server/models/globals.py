import torch
import os

MEM_POOL = torch.cuda.graph_pool_handle()
# This is overridden by the cli
cuda_graphs = os.getenv("CUDA_GRAPHS")
if cuda_graphs is not None and cuda_graphs != "0":
    try:
        cuda_graphs = [int(item) for item in cuda_graphs.split(",")]
    except Exception as e:
        raise RuntimeError(
            f"Could not parse cuda graphs {cuda_graphs}, expected comma separated list for batch sizes to run on: {e}"
        )
else:
    cuda_graphs = None

CUDA_GRAPHS = cuda_graphs
