import time
import os

from datetime import timedelta
from loguru import logger
from pathlib import Path
from typing import Optional, List

from huggingface_hub import file_download, hf_api, HfApi, hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import (
    LocalEntryNotFoundError,
    EntryNotFoundError,
    RevisionNotFoundError,  # noqa # Import here to ease try/except in other part of the lib
)

WEIGHTS_CACHE_OVERRIDE = os.getenv("WEIGHTS_CACHE_OVERRIDE", None)
HF_HUB_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0").lower() in ["true", "1", "yes"]


def _cached_weight_files(model_id: str, revision: Optional[str], extension: str) -> List[str]:
    """Guess weight files from the cached revision snapshot directory"""
    d = _get_cached_revision_directory(model_id, revision)
    if not d:
        return []
    filenames = _weight_files_from_dir(d, extension)
    return filenames


def _weight_hub_files_from_model_info(info: hf_api.ModelInfo, extension: str) -> List[str]:
    return [
        s.rfilename
        for s in info.siblings
        if s.rfilename.endswith(extension)
        and len(s.rfilename.split("/")) == 1
        and "arguments" not in s.rfilename
        and "args" not in s.rfilename
        and "training" not in s.rfilename
    ]


def _weight_files_from_dir(d: Path, extension: str) -> List[str]:
    # os.walk: do not iterate, just scan for depth 1, not recursively
    # see _weight_hub_files_from_model_info, that's also what is
    # done there with the len(s.rfilename.split("/")) == 1 condition
    root, _, files = next(os.walk(str(d)))
    filenames = [f for f in files
                 if f.endswith(extension)
                 and "arguments" not in f
                 and "args" not in f
                 and "training" not in f]
    return filenames


def _get_cached_revision_directory(model_id: str, revision: Optional[str]) -> Optional[Path]:
    if revision is None:
        revision = "main"

    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / Path(
        file_download.repo_folder_name(repo_id=model_id, repo_type="model"))

    if not repo_cache.is_dir():
        # No cache for this model
        return None

    refs_dir = repo_cache / "refs"
    snapshots_dir = repo_cache / "snapshots"

    # Resolve refs (for instance to convert main to the associated commit sha)
    if refs_dir.is_dir():
        revision_file = refs_dir / revision
        if revision_file.exists():
            with revision_file.open() as f:
                revision = f.read()

    # Check if revision folder exists
    if not snapshots_dir.exists():
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    return snapshots_dir / revision


def weight_hub_files(
        model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"
) -> List[str]:
    """Get the weights filenames on the hub"""
    api = HfApi()

    if HF_HUB_OFFLINE:
        filenames = _cached_weight_files(model_id, revision, extension)
    else:
        # Online case, fetch model info from the Hub
        info = api.model_info(model_id, revision=revision)
        filenames = _weight_hub_files_from_model_info(info, extension)

    if not filenames:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id} and revision {revision}.",
            None,
        )

    return filenames


def try_to_load_from_cache(
    model_id: str, revision: Optional[str], filename: str
) -> Optional[Path]:
    """Try to load a file from the Hugging Face cache"""

    d = _get_cached_revision_directory(model_id, revision)
    if not d:
        return None

    # Check if file exists in cache
    cached_file = d / filename
    return cached_file if cached_file.is_file() else None


def weight_files(
    model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"
) -> List[Path]:
    """Get the local files"""
    # Local model
    if Path(model_id).exists() and Path(model_id).is_dir():
        local_files = list(Path(model_id).glob(f"*{extension}"))
        if not local_files:
            raise FileNotFoundError(
                f"No local weights found in {model_id} with extension {extension}"
            )
        return local_files

    try:
        filenames = weight_hub_files(model_id, revision, extension)
    except EntryNotFoundError as e:
        if extension != ".safetensors":
            raise e
        # Try to see if there are pytorch weights
        pt_filenames = weight_hub_files(model_id, revision, extension=".bin")
        # Change pytorch extension to safetensors extension
        # It is possible that we have safetensors weights locally even though they are not on the
        # hub if we converted weights locally without pushing them
        filenames = [
            f"{Path(f).stem.lstrip('pytorch_')}.safetensors" for f in pt_filenames
        ]

    if WEIGHTS_CACHE_OVERRIDE is not None:
        files = []
        for filename in filenames:
            p = Path(WEIGHTS_CACHE_OVERRIDE) / filename
            if not p.exists():
                raise FileNotFoundError(
                    f"File {p} not found in {WEIGHTS_CACHE_OVERRIDE}."
                )
            files.append(p)
        return files

    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(
            model_id, revision=revision, filename=filename
        )
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_id} not found in "
                f"{os.getenv('HUGGINGFACE_HUB_CACHE', 'the local cache')}. "
                f"Please run `text-generation-server download-weights {model_id}` first."
            )
        files.append(cache_file)

    return files


def download_weights(
    filenames: List[str], model_id: str, revision: Optional[str] = None
) -> List[Path]:
    """Download the safetensors files from the hub"""

    def download_file(fname, tries=5, backoff: int = 5):
        local_file = try_to_load_from_cache(model_id, revision, fname)
        if local_file is not None:
            logger.info(f"File {fname} already present in cache.")
            return Path(local_file)

        for idx in range(tries):
            try:
                logger.info(f"Download file: {fname}")
                stime = time.time()
                local_file = hf_hub_download(
                    filename=fname,
                    repo_id=model_id,
                    revision=revision,
                    local_files_only=HF_HUB_OFFLINE,
                )
                logger.info(
                    f"Downloaded {local_file} in {timedelta(seconds=int(time.time() - stime))}."
                )
                return Path(local_file)
            except Exception as e:
                if idx + 1 == tries:
                    raise e
                logger.error(e)
                logger.info(f"Retrying in {backoff} seconds")
                time.sleep(backoff)
                logger.info(f"Retry {idx + 1}/{tries - 1}")

    # We do this instead of using tqdm because we want to parse the logs with the launcher
    start_time = time.time()
    files = []
    for i, filename in enumerate(filenames):
        file = download_file(filename)

        elapsed = timedelta(seconds=int(time.time() - start_time))
        remaining = len(filenames) - (i + 1)
        eta = (elapsed / (i + 1)) * remaining if remaining > 0 else 0

        logger.info(f"Download: [{i + 1}/{len(filenames)}] -- ETA: {eta}")
        files.append(file)

    return files
