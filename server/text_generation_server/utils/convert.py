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


def convert_file(pt_file: Path, st_file: Path):
    """
    Convert a pytorch file to a safetensors file
    """
    logger.info(f"Convert {pt_file} to {st_file}.")

    pt_state = torch.load(pt_file, map_location="cpu")
    if "state_dict" in pt_state:
        pt_state = pt_state["state_dict"]

    remove_shared_pointers(pt_state)

    # Tensors need to be contiguous
    pt_state = {k: v.contiguous() for k, v in pt_state.items()}

    st_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(pt_state, str(st_file), metadata={"format": "pt"})

    # Check that both files are close in size
    check_file_size(pt_file, st_file)

    # Load safetensors state
    st_state = load_file(str(st_file))
    for k in st_state:
        pt_tensor = pt_state[k]
        st_tensor = st_state[k]
        if not torch.equal(pt_tensor, st_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_files(pt_files: List[Path], st_files: List[Path]):
    assert len(pt_files) == len(st_files)

    N = len(pt_files)
    # We do this instead of using tqdm because we want to parse the logs with the launcher
    start = datetime.datetime.now()
    for i, (pt_file, sf_file) in enumerate(zip(pt_files, st_files)):
        elapsed = datetime.datetime.now() - start
        logger.info(f"Convert: [{i + 1}/{N}] -- Took: {elapsed}")
