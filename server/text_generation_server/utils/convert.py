import datetime
import torch
import os

from loguru import logger
from pathlib import Path
from safetensors.torch import save_file, _remove_duplicate_names, load_file
from typing import List


def convert_file(pt_file: Path, sf_file: Path):
    """
    Convert a pytorch file to a safetensors file
    This will remove duplicate tensors from the file.

    Unfortunately, this might not respect *transformers* convention.
    Forcing us to check for potentially different keys during load when looking
    for specific tensors (making tensor sharing explicit).
    """
    loaded = torch.load(pt_file, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    to_removes = _remove_duplicate_names(loaded)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_file)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_file, metadata=metadata)
    reloaded = load_file(sf_file)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
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
