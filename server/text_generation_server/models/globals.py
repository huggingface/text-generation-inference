import torch
import os
from typing import Dict, Optional

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

# NOTE: eventually we should move this into the router and pass back the
# index in all cases.
ADAPTER_TO_INDEX: Optional[Dict[str, int]] = None


def set_adapter_to_index(adapter_to_index: Dict[str, int]):
    global ADAPTER_TO_INDEX
    ADAPTER_TO_INDEX = adapter_to_index


def get_adapter_to_index():
    global ADAPTER_TO_INDEX
    return ADAPTER_TO_INDEX
