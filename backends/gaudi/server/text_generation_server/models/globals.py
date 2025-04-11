import os
from typing import Dict, Optional
from loguru import logger
from text_generation_server.utils.log import log_master

REQUEST_LOGPROBS = os.getenv("REQUEST_LOGPROBS", "0").lower() in {"1", "true"}
ATTENTION = os.getenv("ATTENTION", "default")
# default_prefix_caching = "1" if ATTENTION in {"flashinfer", "flashdecoding"} else "0"
PREFIX_CACHING = os.getenv("PREFIX_CACHING", "0").lower() in {
    "1",
    "true",
}
log_master(logger.info, f"Using prefix caching = {PREFIX_CACHING}")
_expected = {"paged", "default"}
assert (
    ATTENTION in _expected
), f"Attention is not valid {ATTENTION}, expected {_expected}"
log_master(logger.info, f"Using Attention = {ATTENTION}")

TGI_WIGGLE_ROOM = float(os.getenv("TGI_WIGGLE_ROOM", "0.90"))
assert TGI_WIGGLE_ROOM > 0
assert TGI_WIGGLE_ROOM < 1

# This is overridden by the cli
BLOCK_SIZE: int

BLOCK_SIZE = 128


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
