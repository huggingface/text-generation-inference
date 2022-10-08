from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM


def match_suffix(text, suffix):
    return text[-len(suffix) :] == suffix


def shard_model(model_name: str, path: Path, tp_world_size: int, dtype: torch.dtype):
    """BLOOM specific sharding mechanism"""
    save_paths = [
        path / f"{model_name}_tp-rank-{tp_rank}-of-{tp_world_size}.pty"
        for tp_rank in range(tp_world_size)
    ]
    if all(save_path.exists() for save_path in save_paths):
        print("Loading already cached values")
        return save_paths

    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, local_files_only=True
    )

    shards_state_dicts = [{} for _ in range(tp_world_size)]
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    for state_name in keys:
        print(state_name)
        state = state_dict[state_name]
        if any(
            match_suffix(state_name, candidate)
            for candidate in [
                "self_attention.query_key_value.weight",
                "self_attention.query_key_value.bias",
                "mlp.dense_h_to_4h.weight",
                "mlp.dense_h_to_4h.bias",
                "transformer.word_embeddings.weight",
                "lm_head.weight",
            ]
        ):
            output_size = state.shape[0]
            assert output_size % tp_world_size == 0
            block_size = output_size // tp_world_size
            sharded_weights = torch.split(state, block_size, dim=0)
            assert len(sharded_weights) == tp_world_size
            for tp_rank, shard in enumerate(sharded_weights):
                assert shard.shape[0] == block_size
                shards_state_dicts[tp_rank][state_name] = shard.detach().clone()
        elif any(
            match_suffix(state_name, candidate)
            for candidate in [
                "self_attention.dense.weight",
                "mlp.dense_4h_to_h.weight",
                "lm_head.weight",
            ]
        ):
            input_size = state.shape[1]
            assert input_size % tp_world_size == 0
            block_size = input_size // tp_world_size
            sharded_weights = torch.split(state, block_size, dim=1)
            assert len(sharded_weights) == tp_world_size
            for tp_rank, shard in enumerate(sharded_weights):
                assert shard.shape[1] == block_size
                shards_state_dicts[tp_rank][state_name] = shard.detach().clone()
        elif any(
            match_suffix(state_name, candidate)
            for candidate in [
                "self_attention.dense.bias",
                "mlp.dense_4h_to_h.bias",
            ]
        ):
            shards_state_dicts[0][state_name] = state.detach().clone()
            for tp_rank in range(1, tp_world_size):
                shards_state_dicts[tp_rank][state_name] = torch.zeros_like(state)
        else:
            # We duplicate parameters across tp ranks
            for tp_rank in range(tp_world_size):
                shards_state_dicts[tp_rank][state_name] = state.detach().clone()

        del state_dict[state_name]  # delete key from state_dict
        del state  # delete tensor

    # we save state_dict
    for tp_rank, (save_path, shard_state_dict) in enumerate(
        zip(save_paths, shards_state_dicts)
    ):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(shard_state_dict, save_path)
        save_paths.append(save_path)

    return save_paths


if __name__ == "__main__":
    model_name = "bigscience/bloom"
    save_path = Path("/data/shards")
    tp_world_size = 8
    dtype = torch.bfloat16

    shard_model(model_name, save_path, tp_world_size=tp_world_size, dtype=dtype)
