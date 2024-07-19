import copy
from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union
from text_generation_server.utils.merges.utils import (
    calculate_majority_sign_mask,
    disjoint_merge,
    prune,
)
import torch

if TYPE_CHECKING:
    from text_generation_server.adapters.lora import LoraConfig
    from text_generation_server.utils.adapter import ModuleMap


class AdapterParameters:
    def __init__(
        self, adapter_ids, weights, merge_strategy, density, majority_sign_method
    ):
        self.adapter_ids = adapter_ids
        self.weights = weights
        self.merge_strategy = merge_strategy
        self.density = density
        self.majority_sign_method = majority_sign_method


def _apply_weights(
    tensors: Union[torch.Tensor, List[torch.Tensor]], w: torch.Tensor
) -> torch.Tensor:
    if isinstance(tensors, torch.Tensor):
        t = tensors
    else:
        t = torch.stack(tensors, dim=0)

    # element-wise weighting of each task tensor
    # need to unsqueeze weights to match task tensor dimensions
    # for multiplication to apply element-wise
    while len(t.shape) > len(w.shape):
        w = w.unsqueeze(-1)
    return t * w


class MergeStrategy(ABC):
    def merge(
        self, task_tensors: List[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class LinearMerge(MergeStrategy):
    def __init__(self, **kwargs):
        pass

    def merge(
        self, task_tensors: List[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        weighted_task_tensors = _apply_weights(task_tensors, weights)
        return weighted_task_tensors.sum(dim=0)


class TiesMerge(MergeStrategy):
    def __init__(self, density: float, majority_sign_method: str = "total", **kwargs):
        self.density = density
        self.majority_sign_method = majority_sign_method

    def merge(
        self, task_tensors: List[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        # sparsify
        task_tensors = [
            prune(tensor, self.density, method="magnitude") for tensor in task_tensors
        ]
        task_tensors = torch.stack(task_tensors, dim=0)

        # elect sign before applying weights
        majority_sign_mask = calculate_majority_sign_mask(
            task_tensors, method=self.majority_sign_method
        )
        weighted_task_tensors = _apply_weights(task_tensors, weights)

        # disjoint merge
        return disjoint_merge(weighted_task_tensors, majority_sign_mask)


class DareLinearMerge(MergeStrategy):
    def __init__(self, density: float, **kwargs):
        self.density = density

    def merge(
        self, task_tensors: List[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        # sparsify
        task_tensors = [
            prune(tensor, self.density, method="random", rescale=True)
            for tensor in task_tensors
        ]
        weighted_task_tensors = _apply_weights(task_tensors, weights)
        return weighted_task_tensors.sum(dim=0)


class DareTiesMerge(MergeStrategy):
    def __init__(self, density: float, majority_sign_method: str = "total", **kwargs):
        self.density = density
        self.majority_sign_method = majority_sign_method

    def merge(
        self, task_tensors: List[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        # sparsify
        task_tensors = [
            prune(tensor, self.density, method="random", rescale=True)
            for tensor in task_tensors
        ]
        task_tensors = torch.stack(task_tensors, dim=0)

        # elect sign before applying weights
        majority_sign_mask = calculate_majority_sign_mask(
            task_tensors, method=self.majority_sign_method
        )
        weighted_task_tensors = _apply_weights(task_tensors, weights)

        # disjoint merge
        mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
        return mixed_task_tensors


strategy_registry: Dict[str, Type[MergeStrategy]] = {
    "linear": LinearMerge,
    "ties": TiesMerge,
    "dare_linear": DareLinearMerge,
    "dare_ties": DareTiesMerge,
}


def merge_adapters(
    adapters: List[Tuple["ModuleMap", "LoraConfig"]],
    merge_params: AdapterParameters,
) -> Tuple["ModuleMap", "LoraConfig"]:
    # strategy_name = MergeStrategyEnum.Name(merge_params.merge_strategy).lower()
    strategy_name = "linear"

    weights = merge_params.weights
    if not weights:
        weights = torch.ones(len(adapters))
    else:
        weights = torch.tensor(weights)

    merge_config = {
        "density": merge_params.density,
        # "majority_sign_method": MajoritySignMethodEnum.Name(
        #     merge_params.majority_sign_method
        # ).lower(),
        "majority_sign_method": "total",
    }
    merge_strategy = strategy_registry[strategy_name](**merge_config)

    module_maps: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    lora_configs = []
    weight_name_to_adapter_idx = defaultdict(list)

    # input is list of (module_map, lora_config) tuples
    # convert into dict[k][param_name] -> list of tensors
    for idx, (module_map, lora_config) in enumerate(adapters):
        for weight_name, data in module_map.items():
            weight_name_to_adapter_idx[weight_name].append(idx)
            for k, (param_data, param_name) in data.items():
                module_maps[weight_name][k][param_name].append(param_data)
        lora_configs.append(lora_config)

    # validate lora configs are compatible
    _validate_lora_configs(lora_configs)

    # merge tensors for each module such that we have a single ModuleMap:
    # dict[k] -> merged tensor
    merged_module_map: "ModuleMap" = defaultdict(dict)
    for weight_name, data in module_maps.items():
        indices = weight_name_to_adapter_idx[weight_name]
        param_weights = weights[indices]
        for k, param_data in data.items():
            for param_name, tensors in param_data.items():
                merged_tensor = merge_strategy.merge(tensors, param_weights)
                merged_module_map[weight_name][k] = (merged_tensor, param_name)

    # merge lora configs
    merged_lora_config = _merge_lora_configs(lora_configs)

    return merged_module_map, merged_lora_config


def _validate_lora_configs(lora_configs: List["LoraConfig"]):
    # check that all configs have the same rank
    ranks = set(lora_config.r for lora_config in lora_configs)
    if len(ranks) > 1:
        raise ValueError(
            f"unable to merge adapters, lora configs have different ranks: {ranks}"
        )

    if all(len(lora_config.target_modules) == 0 for lora_config in lora_configs):
        raise ValueError(
            "unable to merge adapters, lora configs have no target modules"
        )


def _merge_lora_configs(lora_configs: List["LoraConfig"]) -> "LoraConfig":
    merged_lora_config = copy.copy(lora_configs[0])

    # merge target modules as a union operation
    merged_target_modules = sorted(
        set(
            module
            for lora_config in lora_configs
            for module in lora_config.target_modules
        )
    )
    merged_lora_config.target_modules = merged_target_modules

    return merged_lora_config
