import torch
import os

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import modeling_auto
from huggingface_hub import hf_hub_download, HfApi
from typing import Optional
from pathlib import Path
from typing import Optional, List, Dict
# Needed to properly setup habana_frameworks
import text_generation_server.habana_quantization_env as hq_env

from text_generation_server.utils.speculate import get_speculate, set_speculate
from text_generation_server.models.model import Model
from text_generation_server.models.causal_lm import CausalLM
#from text_generation_server.models.bloom import BLOOM
from text_generation_server.models.starcoder import StarCoder
from text_generation_server.models.vlm_causal_lm import VlmCausalLM
from text_generation_server.models.custom_modeling.llava_next import (
    LlavaNextForConditionalGeneration,
)

from text_generation_server.utils.adapter import (
    AdapterParameters,
    build_layer_weight_lookup,
    load_and_merge_adapters,
    AdapterInfo,
)


from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_id: str,
    lora_adapter_ids: Optional[List[str]],
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    max_input_tokens: int,
) -> Model:
    adapt_transformers_to_gaudi()

    if speculate is not None:
        set_speculate(speculate)
    else:
        set_speculate(0)

    config_dict, _ = PretrainedConfig.get_config_dict(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    model_type = config_dict.get("model_type", None)

    speculator = None
    if "medusa_num_heads" in config_dict:
        medusa_model_id = model_id
        medusa_revision = revision
        model_id = config_dict["base_model_name_or_path"]
        revision = "main"
        speculate_medusa = config_dict["medusa_num_heads"]
        if speculate is not None:
            if speculate > speculate_medusa:
                raise RuntimeError(
                    f"Speculate is set to `{speculate}` but this medusa models only has `{speculate_medusa}` heads, please make them match"
                )
            else:
                set_speculate(speculate)
        else:
            set_speculate(speculate_medusa)

        config_dict, _ = PretrainedConfig.get_config_dict(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        # Reload model type from parent.
        model_type = config_dict.get("model_type", None)
        is_local = Path(medusa_model_id).exists()
        if not is_local:
            medusa_config = hf_hub_download(
                medusa_model_id, revision=medusa_revision, filename="config.json"
            )
            hf_hub_download(
                medusa_model_id,
                revision=medusa_revision,
                filename="medusa_lm_head.safetensors",
            )
            speculator = {
                "path": Path(medusa_config).parent,
                "model_paths": ["medusa_lm_head.safetensors"],
            }
        else:
            speculator = {
                "path": Path(medusa_model_id),
                "model_paths": ["medusa_lm_head.safetensors"],
            }

        method = "medusa"
    elif model_type == "mlp_speculator":
        mlp_model_id = model_id
        mlp_revision = revision
        model_id = config_dict["base_model_name_or_path"]
        revision = "main"
        speculate_mlp = config_dict["n_predict"]
        if speculate is not None:
            if speculate > speculate_mlp:
                raise RuntimeError(
                    f"Speculate is set to `{speculate}` but this mlp_speculator models only has `{speculate_mlp}` heads, please make them match"
                )
            else:
                set_speculate(speculate)
        else:
            set_speculate(speculate_mlp)

        config_dict, _ = PretrainedConfig.get_config_dict(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        # Reload model type from parent.
        model_type = config_dict.get("model_type", None)
        is_local = Path(mlp_model_id).exists()
        extension = ".safetensors"
        if not is_local:
            mlp_speculator_config = hf_hub_download(
                mlp_model_id, revision=mlp_revision, filename="config.json"
            )
            api = HfApi()
            info = api.model_info(mlp_model_id, revision=mlp_revision)
            filenames = [
                s.rfilename
                for s in info.siblings
                if s.rfilename.endswith(extension)
                and len(s.rfilename.split("/")) == 1
                and "arguments" not in s.rfilename
                and "args" not in s.rfilename
                and "training" not in s.rfilename
            ]
            for filename in filenames:
                hf_hub_download(
                    mlp_model_id,
                    revision=mlp_revision,
                    filename=filename,
                )
            speculator_dir_path = Path(mlp_speculator_config).parent
            # if these are downloaded, they get converted to safetensors
            filenames.extend(
                [p for p in os.listdir(speculator_dir_path) if p.endswith(extension)]
            )
            speculator = {
                "path": Path(mlp_speculator_config).parent,
                "model_paths": filenames,
            }
        else:
            speculator = Path(mlp_model_id)
            filenames = [p for p in os.listdir(speculator) if p.endswith(extension)]
            speculator = {"path": speculator, "model_paths": filenames}
        method = "mlp_speculator"
    else:
        method = "n-gram"

    speculate = get_speculate()
    if speculate > 0:
        logger.info(f"Using speculation {method} with {speculate} input ids.")

    model_type = config_dict["model_type"]

    if model_type == "gpt_bigcode":
        return StarCoder(model_id, revision, dtype)

    if model_type == "bloom":
        return BLOOM(
            model_id,
            revision,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "llava_next":
        return VlmCausalLM(
            model_class=LlavaNextForConditionalGeneration,
            model_id=model_id,
            revision=revision,
            quantize=None,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    raise ValueError(f"Unsupported model type {model_type}")


# get_model_with_lora_adapters wraps the internal get_model function and adds support for loading adapters
# this provides a post model loading hook to load adapters into the model after the model has been loaded
def get_model_with_lora_adapters(
    model_id: str,
    lora_adapters: Optional[List[AdapterInfo]],
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    max_input_tokens: int,
    adapter_to_index: Dict[str, int],
):
    lora_adapter_ids = [adapter.id for adapter in lora_adapters]
    model = get_model(
        model_id,
        lora_adapter_ids,
        revision,
        sharded,
        quantize,
        speculate,
        dtype,
        trust_remote_code,
        max_input_tokens,
    )

    if len(lora_adapters) > 0:
        target_to_layer = build_layer_weight_lookup(model.model)

        for index, adapter in enumerate(lora_adapters):
            # The AdapterParameters object allows for merging multiple adapters into a single adapter.
            # At the moment, we only support loading a single adapter into the model, but we keep the
            # AdapterParameters object for easier extension in the future.
            adapter_parameters = AdapterParameters(
                adapter_info=[adapter],
                # when merging multiple adapters we can weight them differently
                # if this is not set, all adapters will be weighted equally
                # see: text_generation_server.utils.merges.strategies for impl
                weights=None,
                merge_strategy=0,
                density=1.0,
                majority_sign_method=0,
            )

            adapter_index = index + 1
            adapter_to_index[adapter.id] = adapter_index

            logger.info(
                f"Loading adapter weights into model: {','.join([adapter.id for adapter in adapter_parameters.adapter_info])}"
            )
            weight_names = tuple([v[0] for v in target_to_layer.values()])
            (
                module_map,
                adapter_config,
                adapter_weight_names,
                adapter_tokenizer,
            ) = load_and_merge_adapters(
                model.model_id,
                adapter_parameters,
                adapter_index,
                weight_names,
                False,
            )

            unused_weight_names = adapter_weight_names.copy()

            adapter_layers = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "qkv_proj",
            ]

            for layer_name in adapter_layers:
                nlayers = (
                    1 if layer_name == "lm_head" else len(model.model.model.layers)
                )
                adapter_weights = LoraWeights.prepare_weights(
                    config=adapter_config,
                    module_map=module_map,
                    layer_type=layer_name,
                    unused_weight_names=unused_weight_names,
                    nlayers=nlayers,
                    dtype=model.dtype,
                    world_size=model.world_size,
                    process_group=model.process_group,
                    target_to_layer=target_to_layer,
                )

                if adapter_weights is None:
                    continue

                model.layer_to_adapter_weights[layer_name].add_adapter(
                    adapter_index, adapter_weights
                )

            if len(unused_weight_names) > 0:
                logger.warning(
                    f"{','.join([a.id for a in lora_adapters])} unused adapter weights: {unused_weight_names}"
                )

            if adapter_tokenizer is not None:
                model.tokenizers.add_tokenizer(adapter_index, adapter_tokenizer)

            model.loaded_adapters.add(adapter_index)

    return model
