import torch
import os
import tempfile
import json

from typing import BinaryIO
from joblib import Parallel, delayed
from functools import partial
from pathlib import Path
from tqdm import tqdm

from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import _request_wrapper, hf_raise_for_status


def match_suffix(text, suffix):
    return text[-len(suffix):] == suffix


def http_get(
        url: str,
        temp_file: BinaryIO,
        *,
        timeout=10.0,
        max_retries=0,
):
    """
    Download a remote file. Do not gobble up errors, and will return errors tailored to the Hugging Face Hub.
    """
    r = _request_wrapper(
        method="GET",
        url=url,
        stream=True,
        timeout=timeout,
        max_retries=max_retries,
    )
    hf_raise_for_status(r)
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            temp_file.write(chunk)


def cache_download_url(url: str, root_dir: Path):
    filename = root_dir / url.split("/")[-1]

    if not filename.exists():
        temp_file_manager = partial(
            tempfile.NamedTemporaryFile, mode="wb", dir=root_dir, delete=False
        )
        with temp_file_manager() as temp_file:
            http_get(url, temp_file)

        os.replace(temp_file.name, filename)
    return filename


def prepare_weights(model_name: str, cache_path: Path, save_path: Path, tp_world_size: int):
    save_paths = [
        save_path / f"{model_name}_tp-rank-{tp_rank}-of-{tp_world_size}.pty"
        for tp_rank in range(tp_world_size)
    ]

    if all(save_path.exists() for save_path in save_paths):
        print("Weights are already prepared")
        return save_paths

    cache_path.mkdir(parents=True, exist_ok=True)
    if model_name == "bigscience/bloom-560m":
        url = hf_hub_url(model_name, filename="pytorch_model.bin")
        cache_download_url(url, cache_path)
    elif model_name == "bigscience/bloom":
        url = hf_hub_url(model_name, filename="pytorch_model.bin.index.json")
        index_path = cache_download_url(url, cache_path)
        with index_path.open("r") as f:
            index = json.load(f)

        # Get unique file names
        weight_files = list(set([filename for filename in index["weight_map"].values()]))
        urls = [hf_hub_url(model_name, filename=filename) for filename in weight_files]

        Parallel(n_jobs=5)(delayed(cache_download_url)(url, cache_path) for url in tqdm(urls))
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    shards_state_dicts = [{} for _ in range(tp_world_size)]

    for weight_path in tqdm(Path(cache_path).glob("*.bin")):
        state_dict = torch.load(weight_path, map_location="cpu")

        keys = list(state_dict.keys())
        for state_name in keys:
            state = state_dict[state_name]
            if any(
                    match_suffix(state_name, candidate)
                    for candidate in [
                        "self_attention.query_key_value.weight",
                        "self_attention.query_key_value.bias",
                        "mlp.dense_h_to_4h.weight",
                        "mlp.dense_h_to_4h.bias",
                        "word_embeddings.weight",
                    ]
            ):
                output_size = state.shape[0]
                assert output_size % tp_world_size == 0
                block_size = output_size // tp_world_size
                sharded_weights = torch.split(state, block_size, dim=0)
                assert len(sharded_weights) == tp_world_size

                for tp_rank, shard in enumerate(sharded_weights):
                    shards_state_dicts[tp_rank]["transformer." + state_name] = shard.detach().clone()

            elif match_suffix(state_name, "lm_head.weight"):
                output_size = state.shape[0]
                assert output_size % tp_world_size == 0
                block_size = output_size // tp_world_size
                sharded_weights = torch.split(state, block_size, dim=0)
                assert len(sharded_weights) == tp_world_size

                for tp_rank, shard in enumerate(sharded_weights):
                    shards_state_dicts[tp_rank][state_name] = shard.detach().clone()

            elif any(
                    match_suffix(state_name, candidate)
                    for candidate in [
                        "self_attention.dense.weight",
                        "mlp.dense_4h_to_h.weight",
                    ]
            ):
                input_size = state.shape[1]
                assert input_size % tp_world_size == 0
                block_size = input_size // tp_world_size
                sharded_weights = torch.split(state, block_size, dim=1)
                assert len(sharded_weights) == tp_world_size
                for tp_rank, shard in enumerate(sharded_weights):
                    shards_state_dicts[tp_rank]["transformer." + state_name] = shard.detach().clone()

            elif any(
                    match_suffix(state_name, candidate)
                    for candidate in [
                        "self_attention.dense.bias",
                        "mlp.dense_4h_to_h.bias",
                    ]
            ):
                shards_state_dicts[0]["transformer." + state_name] = state.detach().clone()
                for tp_rank in range(1, tp_world_size):
                    shards_state_dicts[tp_rank]["transformer." + state_name] = torch.zeros_like(state)

            else:
                # We duplicate parameters across tp ranks
                for tp_rank in range(tp_world_size):
                    shards_state_dicts[tp_rank]["transformer." + state_name] = state.detach().clone()

            del state_dict[state_name]  # delete key from state_dict
            del state  # delete tensor
        del state_dict

    # we save state_dict
    for tp_rank, (save_path, shard_state_dict) in enumerate(
            zip(save_paths, shards_state_dicts)
    ):
        save_paths.append(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            print(f"Skipping {save_path} as it already exists")
        else:
            torch.save(shard_state_dict, save_path)

    return save_paths


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--cache-path", required=True, type=str)
    parser.add_argument("--save-path", required=True, type=str)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()

    prepare_weights(args.model_name, Path(args.cache_path), Path(args.save_path), args.world_size)
