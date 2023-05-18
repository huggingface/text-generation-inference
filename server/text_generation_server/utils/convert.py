import concurrent
import time
import datetime
import torch

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import timedelta
from loguru import logger
from pathlib import Path
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from typing import Dict, List


def check_file_size(source_file: Path, target_file: Path):
    """
    Check that two files are close in size
    """
    source_file_size = source_file.stat().st_size
    target_file_size = target_file.stat().st_size

    if (source_file_size - target_file_size) / source_file_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {source_file}: {source_file_size}
         - {target_file}: {target_file_size}
         """
        )


def remove_shared_pointers(tensors: Dict[str, torch.Tensor]):
    """
    For a Dict of tensors, check if two or more tensors point to the same underlying memory and
    remove them
    """
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)

    # Iterate over all found memory addresses
    for ptr, names in ptrs.items():
        if len(names) > 1:
            # Multiple tensors are point to the same memory
            # Only keep the first tensor
            for name in names[1:]:
                tensors.pop(name)


def convert_file(pt_file: Path, sf_file: Path):
    """
    Convert a pytorch file to a safetensors file
    """
    logger.info(f"Convert {pt_file} to {sf_file}.")

    pt_state = torch.load(pt_file, map_location="cpu")
    if "state_dict" in pt_state:
        pt_state = pt_state["state_dict"]

    remove_shared_pointers(pt_state)

    # Tensors need to be contiguous
    pt_state = {k: v.contiguous() for k, v in pt_state.items()}

    sf_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(pt_state, str(sf_file), metadata={"format": "pt"})

    # Check that both files are close in size
    check_file_size(pt_file, sf_file)

    # Load safetensors state
    for k in pt_state:
        pt_tensor = pt_state[k]
        with safe_open(sf_file, framework="pt") as f:
            sf_tensor = f.get_tensor(k)
            if not torch.equal(pt_tensor, sf_tensor):
                raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_files(pt_files: List[Path], sf_files: List[Path]):
    assert len(pt_files) == len(sf_files)

    N = len(pt_files)
    # We do this instead of using tqdm because we want to parse the logs with the launcher

    for i, (pt_file, sf_file) in enumerate(zip(pt_files, sf_files)):
        start = datetime.datetime.now()
        convert_file(pt_file, sf_file)
        elapsed = datetime.datetime.now() - start
        logger.info(f"Convert: [{i + 1}/{N}] -- Took: {elapsed}")
