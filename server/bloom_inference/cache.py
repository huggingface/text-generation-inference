from bloom_inference.model import Batch
from typing import Dict, Optional


class Cache:
    def __init__(self):
        self.cache: Dict[int, Batch] = {}

    def pop(self, batch_id: int) -> Optional[Batch]:
        return self.cache.pop(batch_id, None)

    def set(self, entry: Batch):
        if entry is not None:
            self.cache[entry.batch_id] = entry

    def delete(self, batch_id: int):
        del self.cache[batch_id]

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache.keys())
