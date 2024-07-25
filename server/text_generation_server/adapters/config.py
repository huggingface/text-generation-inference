# Origin:   https://github.com/predibase/lorax
# Path:     lorax/server/lorax_server/adapters/config.py
# License:  Apache License Version 2.0, January 2004

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Set, Tuple

import torch

from text_generation_server.adapters.weights import AdapterWeights

if TYPE_CHECKING:
    from text_generation_server.models.model import Model


@dataclass
class ModuleMap:
    module_name: str
    module_weights: Dict[str, Tuple[torch.Tensor, str]]


@dataclass
class AdapterConfig(ABC):
    base_model_name_or_path: str

    @abstractmethod
    def map_weights_for_model(
        self,
        adapter_weights: Dict[int, AdapterWeights],
        weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        pass
