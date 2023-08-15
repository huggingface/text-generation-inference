import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple

import torch
from loguru import logger
from peft import LoraConfig
from peft.utils import transpose
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from text_generation_server.utils.hub import weight_files


def compute_delta_weight(
    lora_A: torch.Tensor, 
    lora_B: torch.Tensor, 
    fan_in_fan_out: bool, 
    alpha: float, 
    r: float
) -> torch.Tensor:
    """Computes the delta weight for a Linear layer given A and B LoRA matrices.
    
    TODO: add logic for other module types beyond Linear layers.
    
    Reference: https://github.com/huggingface/peft/blob/v0.4.0/src/peft/tuners/lora.py#L799-L806
    """
    scaling = alpha / r
    delta_weight = transpose(lora_B @ lora_A, fan_in_fan_out) * scaling
    return delta_weight


def merge_adapter_weights(
    model_weights: Dict[str, torch.Tensor], 
    adapter_weights: Dict[str, torch.Tensor], 
    adapter_config: LoraConfig
) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
    """Merges the adapter weights into the model weights."""
    module_mapping = defaultdict(dict)
    processed_adapter_weight_names = set()

    # map the original tensor names to their adapter counterparts
    for weight_name in model_weights:
        end_idx = weight_name.rfind(".weight")
        key = weight_name[:end_idx]
        for adapter_weight_name in adapter_weights:
            if key in adapter_weight_name:
                # example value: 'base_model.model.model.layers.10.self_attn.v_proj.lora_B.weight'
                # matrix_type gets the second to last element in the module name, i.e. 'lora_B'
                matrix_type = adapter_weight_name.split(".")[-2]
                module_mapping[weight_name][matrix_type] = adapter_weight_name
                processed_adapter_weight_names.add(adapter_weight_name)
    
    # merge adapter weights into model weights
    merged_weights = {}
    for weight_name, adapter_weight_names in tqdm(
        module_mapping.items(), desc="Merging adapter weights", total=len(module_mapping)):

        # TODO: support adapter types beyond LoRA
        lora_A = adapter_weights[adapter_weight_names["lora_A"]]
        lora_B = adapter_weights[adapter_weight_names["lora_B"]]
        delta_weight = compute_delta_weight(
            lora_A, lora_B, adapter_config.fan_in_fan_out, adapter_config.lora_alpha, adapter_config.r)
        merged_weights[weight_name] = model_weights[weight_name] + delta_weight
    return merged_weights, processed_adapter_weight_names


def create_merged_weight_files(
    adapter_id: str, 
    model_id: str,
    model_weight_filenames: List[Path]
) -> List[Path]:
    """Creates merged weight files for the given adapter ID and filenames."""
    adapter_filenames = weight_files(adapter_id, extension=".safetensors")
    
    adapter_config = LoraConfig.from_pretrained(adapter_id)
    if adapter_config.base_model_name_or_path != model_id:
        raise ValueError(f"Adapter {adapter_id} is not compatible with model {model_id}")
    
    # load adapter weights from all shards (should have relatively small memory footprint)
    adapter_weights = {}
    for filename in adapter_filenames:
        adapter_weights.update(load_file(filename))
    remaining_adapter_weight_names = set(adapter_weights.keys())

    merged_weight_directory = f"/data/{adapter_id.replace('/', '--')}-merged/"
    # just grab the existing files if they already exist and return immediately
    if os.path.exists(merged_weight_directory):
        logger.info("Merged weight files already exist, skipping merge computation.")
        return weight_files(merged_weight_directory)

    os.makedirs(merged_weight_directory)
    merged_weight_filenames = []
    for filename in model_weight_filenames:
        model_weights = load_file(filename)
        merged_weights, processed_adapter_weight_names = merge_adapter_weights(
            model_weights, adapter_weights, adapter_config)
        
        merged_adapter_filename = Path(merged_weight_directory, os.path.basename(filename))
        save_file(merged_weights, merged_adapter_filename)
        logger.debug(f"Saved merged weights into {merged_adapter_filename}")

        merged_weight_filenames.append(merged_adapter_filename)
        remaining_adapter_weight_names = remaining_adapter_weight_names.difference(
            processed_adapter_weight_names)
    
    if len(remaining_adapter_weight_names) > 0:
        logger.warning("WARNING: The following lora weights were not merged into the model weights:")
        for lora_name in remaining_adapter_weight_names:
            logger.warning("\t" + lora_name)

    return merged_weight_filenames


def main():
    adapter_id = "arnavgrg/codealpaca-qlora"
    adapter_config = LoraConfig.from_pretrained(adapter_id)
    model_id = adapter_config.base_model_name_or_path
    model_weight_filenames = weight_files(model_id, extension=".safetensors")
    
    merged_adapter_filenames = create_merged_weight_files(adapter_id, model_id, model_weight_filenames)
    print(merged_adapter_filenames)


if __name__ == '__main__':
    main()