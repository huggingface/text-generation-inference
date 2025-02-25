# Origin:   https://github.com/predibase/lorax
# Path:     lorax/server/lorax_server/utils/adapter.py
# License:  Apache License Version 2.0, January 2004

import warnings
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Set, Tuple, Optional, List

from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from text_generation_server.utils.merges.strategies import merge_adapters

from text_generation_server.utils import hub
from text_generation_server.adapters.lora import LoraConfig


if TYPE_CHECKING:
    from text_generation_server.adapters.config import AdapterConfig, ModuleMap


BASE_MODEL_ADAPTER_ID = "__base_model__"


@dataclass
class AdapterInfo:
    id: str
    path: Optional[str]
    revision: Optional[str] = None


@dataclass
class AdapterParameters:
    adapter_info: Tuple[AdapterInfo]
    weights: Tuple[float]
    merge_strategy: NotImplemented
    density: float
    majority_sign_method: NotImplemented


@dataclass
class AdapterSource:
    adapter_id: str
    model_id: str
    revision: str


def parse_lora_adapters(lora_adapters: Optional[str]) -> List[AdapterInfo]:
    if not lora_adapters:
        return []

    adapter_list = []
    for adapter in lora_adapters.split(","):
        adapter = adapter.strip()
        if adapter.count("=") > 1 or adapter.count("@") > 1:
            raise ValueError(f"Invalid LoRA adapter format: {adapter}")
        match = re.match(r"^([^=@]+)(?:=([^@]+))?(?:@(.+))?$", adapter)

        if match:
            adapter_id, path, revision = match.groups()
            adapter_list.append(
                AdapterInfo(id=adapter_id, path=path, revision=revision)
            )
        else:
            raise ValueError(f"Invalid LoRA adapter format: {adapter}")
    return adapter_list


def load_and_merge_adapters(
    model_id: str,
    adapter_parameters: AdapterParameters,
    adapter_index: int,
    weight_names: Tuple[str],
    trust_remote_code: bool = False,
) -> Tuple["ModuleMap", "AdapterConfig", Set[str], PreTrainedTokenizer]:
    if len(adapter_parameters.adapter_info) == 1:
        adapter = next(iter(adapter_parameters.adapter_info))
        return load_module_map(
            model_id,
            adapter.revision,
            adapter.id,
            adapter.path,
            weight_names,
            trust_remote_code,
        )

    adapter_params = AdapterParametersContainer(adapter_parameters, adapter_index)
    return _load_and_merge(
        model_id,
        adapter_params,
        weight_names,
        trust_remote_code,
    )


@dataclass
class AdapterParametersContainer:
    adapter_parameters: AdapterParameters
    adapter_index: int

    def __hash__(self) -> int:
        return self.adapter_index


@lru_cache(maxsize=32)
def _load_and_merge(
    model_id: str,
    adapter_params: AdapterParametersContainer,
    weight_names: Tuple[str],
    trust_remote_code: bool = False,
) -> Tuple["ModuleMap", "AdapterConfig", Set[str], PreTrainedTokenizer]:
    params = adapter_params.adapter_parameters

    adapters_to_merge = []
    merged_weight_names = set()
    tokenizer = None
    for adapter in params.adapter_info:
        if adapter.id == BASE_MODEL_ADAPTER_ID:
            raise ValueError("Base model adapter cannot be merged.")

        module_map, adapter_config, adapter_weight_names, adapter_tokenizer = (
            load_module_map(
                model_id,
                adapter.revision,
                adapter.id,
                adapter.path,
                weight_names,
                trust_remote_code,
            )
        )

        adapters_to_merge.append((module_map, adapter_config))
        merged_weight_names = merged_weight_names.union(adapter_weight_names)
        if tokenizer is None:
            tokenizer = adapter_tokenizer

    if len(adapters_to_merge) == 0:
        raise ValueError("No adapters to merge.")

    module_map, adapter_config = merge_adapters(adapters_to_merge, params)
    return module_map, adapter_config, merged_weight_names, tokenizer


def check_architectures(
    model_id: str,
    adapter_id: str,
    adapter_config: "AdapterConfig",
    trust_remote_code: bool = False,
):
    try:
        if not adapter_config.base_model_name_or_path:
            # Avoid execution latency caused by the network connection retrying for AutoConfig.from_pretrained(None)
            return

        expected_config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        model_config = AutoConfig.from_pretrained(
            adapter_config.base_model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception as e:
        warnings.warn(
            f"Unable to check architecture compatibility for adapter '{adapter_id}' "
            f"against model '{model_id}'. Assuming they are compatible. Error: {e}"
        )
        return

    if model_config.architectures == expected_config.architectures:
        warnings.warn(
            f"Adapter '{adapter_id}' was not trained on base model '{model_id}'. "
            f"If you encounter issues, use --model-id '{adapter_config.base_model_name_or_path}' instead."
        )
    else:
        # TODO(travis): revisit this when we support clasification heads which will not use CausalLM
        raise ValueError(
            f"Adapter '{adapter_id}' is not compatible with model '{model_id}'. "
            f"Architectures differ: {model_config.architectures} != {expected_config.architectures}. "
            f"Use --model-id '{adapter_config.base_model_name_or_path}' instead."
        )


@lru_cache(maxsize=128)
def load_module_map(
    model_id: str,
    revision: str,
    adapter_id: str,
    adapter_path: Optional[str],
    weight_names: Tuple[str],
    trust_remote_code: bool = False,
) -> Tuple["ModuleMap", "AdapterConfig", Set[str], PreTrainedTokenizer]:
    adapter_config = LoraConfig.load(adapter_path or adapter_id, None)

    if not adapter_path and adapter_config.base_model_name_or_path != model_id:
        check_architectures(model_id, adapter_id, adapter_config, trust_remote_code)

    adapter_filenames = (
        hub._weight_files_from_dir(adapter_path, extension=".safetensors")
        if adapter_path
        else hub._cached_weight_files(
            adapter_id, revision=revision, extension=".safetensors"
        )
    )

    # throw an error if no adapter weights are found
    if not adapter_filenames:
        raise FileNotFoundError(
            f"No adapter weights found for adapter '{adapter_id}' and revision '{revision}'."
        )

    try:
        adapter_tokenizer = AutoTokenizer.from_pretrained(
            adapter_config.config_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        # Adapter does not have a tokenizer, so fallback to base model tokenizer
        adapter_tokenizer = None

    # load adapter weights from all shards (should have relatively small memory footprint)
    adapter_weights = {}
    for filename in adapter_filenames:
        adapter_weights.update(load_file(filename))

    # map the model weights to the relevant adapter weights (LoRA A and B matrices)
    module_map, adapter_weight_names = adapter_config.map_weights_for_model(
        adapter_weights, weight_names
    )
    return module_map, adapter_config, adapter_weight_names, adapter_tokenizer


def get_attn_weights(i, layer):
    qkv = layer.self_attn.query_key_value
    weights = {}

    for k in ["q", "k", "v"]:
        key = (i, f"{k}_proj")
        value = (f"model.layers.{i}.self_attn.{k}_proj", qkv)
        weights[key] = value

    # also add the qkv_proj weight for the adapter
    weights[(i, "qkv_proj")] = (
        f"model.layers.{i}.self_attn.qkv_proj",
        qkv,
    )

    weights[(i, "o_proj")] = (
        f"model.layers.{i}.self_attn.o_proj",
        layer.self_attn.o_proj,
    )

    return weights


def get_mlp_weights(i, layer):
    weights = {}
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "gate_up_proj"):
            # handle combined gate_up_proj (e.g., for some LLaMA variants)
            weights.update(
                {
                    (i, "gate_proj"): (
                        f"model.layers.{i}.mlp.gate_proj",
                        mlp.gate_up_proj,
                    ),
                    (i, "up_proj"): (f"model.layers.{i}.mlp.up_proj", mlp.gate_up_proj),
                }
            )
        else:
            # handle separate gate_proj, up_proj, and down_proj (e.g., for Gemma)
            if hasattr(mlp, "gate_proj"):
                weights[(i, "gate_proj")] = (
                    f"model.layers.{i}.mlp.gate_proj",
                    mlp.gate_proj,
                )
            if hasattr(mlp, "up_proj"):
                weights[(i, "up_proj")] = (f"model.layers.{i}.mlp.up_proj", mlp.up_proj)

        if hasattr(mlp, "down_proj"):
            weights[(i, "down_proj")] = (
                f"model.layers.{i}.mlp.down_proj",
                mlp.down_proj,
            )

    return weights


# build_layer_weight_lookup creates a mapping of model layers to their corresponding
# weight tensors and paths. It builds a dictionary that maps layer identifiers to tuples
# containing the weight tensor path and the actual layer object. This mapping is needed
# for the lora adapter to know which weights to update when applying the adapter.
def build_layer_weight_lookup(model):
    if hasattr(model, "language_model"):
        m = model.language_model.model
    elif hasattr(model, "text_model"):
        m = model.text_model.model
    else:
        m = model.model

    layer_weights = {}

    for i, layer in enumerate(m.layers):
        attn_weights = get_attn_weights(i, layer)
        mlp_weights = get_mlp_weights(i, layer)

        layer_weights.update(attn_weights)
        layer_weights.update(mlp_weights)

    lm_head = None
    if hasattr(m, "lm_head"):
        lm_head = m.lm_head
    elif hasattr(model, "lm_head"):
        lm_head = model.lm_head

    if lm_head:
        layer_weights[(0, "lm_head")] = ("lm_head", lm_head)

    return layer_weights
