from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from text_generation_server.adapters.weights import AdapterWeights

if TYPE_CHECKING:
    from text_generation_server.models.model import Model


ModuleMap = Dict[str, Dict[str, Tuple[torch.Tensor, str]]]


@dataclass
class AdapterConfig(ABC):
    base_model_name_or_path: str

    @abstractmethod
    def map_weights_for_model(
        self,
        adapter_weights: Dict,
        weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        pass

    @abstractmethod
    def load_batched_adapter_weights(
        self,
        model: "Model",
        module_map: Dict[str, Dict],
        layer_type: str,
        unused_weight_names: Set[str],
        dynamic: bool,
    ) -> Optional[AdapterWeights]:
        pass
