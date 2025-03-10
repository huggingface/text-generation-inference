import importlib

from loguru import logger
from kernels import load_kernel as hf_load_kernel

from text_generation_server.utils.log import log_once


def load_kernel(*, module: str, repo_id: str):
    """
    Load a kernel. First try to load it as the given module (e.g. for
    local development), falling back to a locked Hub kernel.
    """
    try:
        m = importlib.import_module(module)
        log_once(logger.info, f"Using local module for `{module}`")
        return m
    except ModuleNotFoundError:
        return hf_load_kernel(repo_id=repo_id)


__all__ = ["load_kernel"]
