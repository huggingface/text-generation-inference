# Origin:   https://github.com/predibase/lorax
# Path:     lorax/server/lorax_server/adapters/weights.py
# License:  Apache License Version 2.0, January 2004

from abc import ABC, abstractclassmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Type

import torch


@dataclass
class AdapterBatchMetadata:
    # [batch_size]
    adapter_indices: torch.Tensor

    # [num_adapters]
    adapter_set: Set[int]

    # [num_segments + 1]
    adapter_segments: torch.Tensor

    # [num_segments]
    # maps from segment index to adapter index, i.e.:
    # segment_indices[s] == adapter_indices[i]
    segment_indices: List[int]


class AdapterWeights(ABC):
    @abstractclassmethod
    def get_batch_types(cls) -> List[Type["BatchAdapterWeights"]]:
        pass

    @property
    def speculative_tokens(self) -> int:
        return 0


class BatchAdapterWeights(ABC):
    @abstractclassmethod
    def has_adapter(self, adapter_index: int) -> bool:
        pass

    @abstractclassmethod
    def key(cls) -> str:
        pass

    @abstractclassmethod
    def load(
        cls,
        adapter_weights: Dict[int, AdapterWeights],
        meta: "AdapterBatchMetadata",
        prefill: bool,
        prefill_head_indices: torch.Tensor,
    ) -> Optional["BatchAdapterWeights"]:
        pass


class LayerAdapterWeights:
    """Adapter weights that apply to a particular layer."""

    def __init__(self):
        self.adapter_weights: Dict[int, AdapterWeights] = {}

    def add_adapter(self, adapter_idx: int, weights: AdapterWeights):
        self.adapter_weights[adapter_idx] = weights

    def remove_adapter(self, adapter_idx: int):
        if adapter_idx not in self.adapter_weights:
            return
        del self.adapter_weights[adapter_idx]

    @property
    def max_speculative_tokens(self) -> int:
        return max(
            adapter_weights.speculative_tokens
            for adapter_weights in self.adapter_weights.values()
        )

    def is_empty(self) -> bool:
        return len(self.adapter_weights) == 0

    def get_data(
        self,
        meta: AdapterBatchMetadata,
        prefill: bool,
        prefill_head_indices: Optional[torch.Tensor],
    ) -> Dict[str, BatchAdapterWeights]:
        # bucket adapters by batch class
        adapter_batch_types: Dict[
            Type[BatchAdapterWeights], Dict[int, AdapterWeights]
        ] = defaultdict(dict)
        for adapter_index, adapter_weights in self.adapter_weights.items():
            for batch_type in adapter_weights.get_batch_types():
                adapter_batch_types[batch_type][adapter_index] = adapter_weights

        batch_data = {}
        for batch_type, adapter_weights in adapter_batch_types.items():
            batched_weights = batch_type.load(
                adapter_weights, meta, prefill, prefill_head_indices
            )
            if batched_weights is not None:
                batch_data[batch_type.key()] = batched_weights
        return batch_data


@dataclass
class AdapterBatchData:
    meta: AdapterBatchMetadata

    # layer type -> adapter type -> batch weight data
    data: Dict[str, Dict[str, BatchAdapterWeights]]

    prefill: bool

    @staticmethod
    def from_meta(
        meta: AdapterBatchMetadata,
        weights: Dict[str, LayerAdapterWeights],
        prefill: bool,
        prefill_head_indices: Optional[torch.Tensor],
    ) -> "AdapterBatchData":
        data = {}
        for k, v in weights.items():
            if v.is_empty():
                continue
            data[k] = v.get_data(
                meta, prefill, prefill_head_indices if k == "lm_head" else None
            )
        return AdapterBatchData(meta=meta, data=data, prefill=prefill)

    def ranks(self) -> Set[int]:
        # TODO(travis): refactor to be less coupled to lora implementation
        ranks = set()
        for layer_data in self.data.values():
            lora_data = layer_data.get("lora")
            if lora_data is None:
                continue

            for rank_data in lora_data.rank_data.values():
                ranks.add(rank_data.rank)

        return ranks

    def layer_names(self) -> Set[str]:
        return set(self.data.keys())

    def adapter_keys(self) -> Set[str]:
        adapter_keys = set()
        for layer_data in self.data.values():
            adapter_keys.update(layer_data.keys())
        return adapter_keys

    @property
    def max_rank(self) -> int:
        ranks = self.ranks()
        return max(ranks) if len(ranks) > 0 else 0
