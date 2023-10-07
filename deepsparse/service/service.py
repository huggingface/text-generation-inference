from typing import Dict, List, Tuple
from service.causal_lm import DeepSparseCausalLM, DeepSparseCausalLMBatch
from utils import Generation, CachedBatch, Batch

class BatchCache:
    def __init__(self):
        self.cache: Dict[int, DeepSparseCausalLMBatch] = {}

    def pop(self, batch_id: int) -> DeepSparseCausalLMBatch:
        batch = self.cache.pop(batch_id, None)
        assert batch is not None, "Batch ID {batch_id} not found in cache."
        return batch

    def set(self, entry: DeepSparseCausalLMBatch):
        if entry is not None:
            self.cache[entry.batch_id] = entry

    def delete(self, batch_id: int):
        batch = self.pop(batch_id)
        if batch is not None:
            del batch

    def clear(self):
        keys = list(self.cache.keys())
        for k in keys:
            self.delete(k)

    def __len__(self):
        return len(self.cache.keys())

class DeepSparseService:
    def __init__(
        self, 
        model: DeepSparseCausalLM
    ):
        self.model = model
        self.cache = BatchCache()

    def ClearCache(self):
        self.cache.clear()

    def FilterBatch(self, batch_id: int, request_ids: List[int]) -> CachedBatch:
        ds_batch = self.cache.pop(batch_id)
        filtered_ds_batch = ds_batch.filter(request_ids)
        self.cache.set(filtered_ds_batch)

        return filtered_ds_batch.to_cached_batch()

    def Prefill(self, batch: Batch) -> Tuple[Generation, CachedBatch]:
        ds_batch = DeepSparseCausalLMBatch.from_batch(
            batch=batch,
            tokenizer=self.model.tokenizer
        )

        generations, next_ds_batch = self.model.generate_token(ds_batch)
        assert len(generations) == 1
        self.cache.set(next_ds_batch)

        return generations[0], (next_ds_batch.to_cached_batch() if next_ds_batch else None)

    def Decode(self, batches: List[CachedBatch]) -> Tuple[List[Generation], CachedBatch]:
        assert len(batches) != 0, "Must provide at least one batch"

        ds_batches = []
        for cached_batch in batches:
            ds_batches.append(self.cache.pop(cached_batch.batch_id))

        if len(ds_batches) > 1:
            ds_batch = DeepSparseCausalLMBatch.concatenate(ds_batches)
        else:
            ds_batch = ds_batches[0]

        generations, next_ds_batch = self.model.generate_token(ds_batch)
        self.cache.set(next_ds_batch)

        return generations, (next_ds_batch.to_cached_batch() if next_ds_batch else None)
