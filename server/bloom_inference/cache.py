import torch

from dataclasses import dataclass
from typing import Dict, Optional, List

from bloom_inference.pb import generate_pb2
from bloom_inference.utils import NextTokenChooser, StoppingCriteria


@dataclass
class CacheEntry:
    batch_id: int
    request_ids: List[int]
    input_ids: Dict[str, torch.Tensor]
    all_input_ids: List[torch.Tensor]
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    def __len__(self):
        return len(self.request_ids)

    def to_pb(self):
        return generate_pb2.CacheEntry(
            id=self.batch_id,
            request_ids=self.request_ids,
            sequence_length=max(len(entry) for entry in self.all_input_ids),
        )


class Cache:
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}

    def pop(self, batch_id: str) -> Optional[CacheEntry]:
        return self.cache.pop(batch_id, None)

    def set(self, entry: CacheEntry):
        if entry is not None:
            self.cache[entry.batch_id] = entry

    def delete(self, batch_id: str):
        del self.cache[batch_id]

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache.keys())
