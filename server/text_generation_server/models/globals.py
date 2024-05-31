import torch
import os
from loguru import logger

MEM_POOL = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
# This is overridden by the cli
cuda_graphs = os.getenv("CUDA_GRAPHS")
if cuda_graphs is not None:
    try:
        cuda_graphs = [int(item) for item in cuda_graphs.split(",")]
    except Exception as e:
        raise RuntimeError(
            f"Could not parse cuda graphs {cuda_graphs}, expected comma separated list for batch sizes to run on: {e}"
        )
else:
    cuda_graphs = None

CUDA_GRAPHS = cuda_graphs

# This is overridden at model loading.
global MODEL_ID
MODEL_ID = None


def set_model_id(model_id: str):
    global MODEL_ID
    MODEL_ID = model_id
