import torch
import os
from loguru import logger
from typing import Dict, Optional

from text_generation_server.utils.log import log_master

REQUEST_LOGPROBS = os.getenv("REQUEST_LOGPROBS", "0").lower() in {"1", "true"}
ATTENTION = os.environ["ATTENTION"]
# default_prefix_caching = "1" if ATTENTION in {"flashinfer", "flashdecoding"} else "0"
PREFIX_CACHING = os.environ["PREFIX_CACHING"].lower() in {
    "1",
    "true",
}
PREFILL_CHUNKING = os.getenv("PREFILL_CHUNKING", "1").lower() in {"1", "true"}
log_master(logger.info, f"Using prefix caching = {PREFIX_CACHING}")
_expected = {"paged", "flashdecoding", "flashdecoding-ipex", "flashinfer"}
assert (
    ATTENTION in _expected
), f"Attention is not valid {ATTENTION}, expected {_expected}"
log_master(logger.info, f"Using Attention = {ATTENTION}")

if PREFIX_CACHING and ATTENTION not in {
    "flashinfer",
    "flashdecoding",
    "flashdecoding-ipex",
}:
    raise RuntimeError("Prefix caching is only supported with flashinfer")

MEM_POOL = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
# Test a 70B model on 4xA100 under load for latest failure
TGI_WIGGLE_ROOM = float(os.getenv("TGI_WIGGLE_ROOM", "0.90"))
assert TGI_WIGGLE_ROOM > 0
assert TGI_WIGGLE_ROOM < 1


# This is overridden by the cli
BLOCK_SIZE: int
if ATTENTION == "flashdecoding":
    BLOCK_SIZE = 256
elif ATTENTION == "flashinfer":
    BLOCK_SIZE = 1
elif ATTENTION == "flashdecoding-ipex":
    BLOCK_SIZE = 64
else:
    BLOCK_SIZE = 16

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
# sorting the cuda graphs in descending order helps reduce the
# memory impact and results in less memory usage
if cuda_graphs is not None:
    cuda_graphs.sort(reverse=True)

CUDA_GRAPHS = cuda_graphs

# NOTE: eventually we should move this into the router and pass back the
# index in all cases.
ADAPTER_TO_INDEX: Optional[Dict[str, int]] = None


def set_adapter_to_index(adapter_to_index: Dict[str, int]):
    global ADAPTER_TO_INDEX
    ADAPTER_TO_INDEX = adapter_to_index


def get_adapter_to_index():
    global ADAPTER_TO_INDEX
    return ADAPTER_TO_INDEX
