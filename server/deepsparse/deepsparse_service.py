from typing import Optional, Dict, List
from server.deepsparse.deepsparse_causal_lm import (
    DeepSparseCausalLM, DeepSparseCausalLMBatch
)
from server.deepsparse.deepsparse_requests import (
    PrefillRequest, DecodeRequest, FilterBatchRequest, 
    Generation, CachedBatch
)

class BatchCache:
    def __init__(self):
        self.cache: Dict[int, DeepSparseCausalLMBatch] = {}

    def pop(self, batch_id: int) -> Optional[DeepSparseCausalLMBatch]:
        return self.cache.pop(batch_id, None)

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

    def FilterBatch(
        self, 
        request: FilterBatchRequest
    ) -> CachedBatch:
        ds_batch = self.cache.pop(request.batch_id)
        assert ds_batch is not None, "Batch ID {request.batch_id} not found in cache."
        filtered_batch = ds_batch.filter(request.request_ids)
        self.cache.set(filtered_batch)

        return filtered_batch.to_batch()

    def Prefill(
        self, 
        request: PrefillRequest
    ) -> [Generation, CachedBatch]:
        ds_batch = DeepSparseCausalLMBatch.from_batch(
            batch=request.batch,
            tokenizer=self.model.tokenizer
        )

        generations, next_ds_batch = self.model.generate_token(ds_batch)
        assert len(generations) == 1
        self.cache.set(next_ds_batch)

        return generations[0], next_ds_batch.to_batch()

    def Decode(
        self, 
        request: DecodeRequest
    ) -> [List[Generation], CachedBatch]:
        assert len(request.batches) != 0, "Must provide at least one batch"

        ds_batches = []
        for batch in request.batches:
            ds_batch = self.cache.pop(batch.batch_id)
            assert batch is not None, "Batch ID {batch.id} not found in cache."
            ds_batches.append(ds_batch)

        if len(ds_batches) > 1:
            ds_batch = DeepSparseCausalLMBatch.concatenate(ds_batches)
        else:
            ds_batch = ds_batches[0]

        generations, next_ds_batch = self.model.generate_token(ds_batch)
        self.cache.set(next_ds_batch)

        return generations, next_ds_batch.to_batch() if next_ds_batch else None