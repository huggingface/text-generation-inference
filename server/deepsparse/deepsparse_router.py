from threading import Condition
from typing import List, Dict, Optional

from server.deepsparse.deepsparse_service import DeepSparseService
from server.deepsparse.deepsparse_requests import (
    CachedBatch, Batch, Generation,
    PrefillRequest, DecodeRequest, FilterBatchRequest, 
)
from server.deepsparse.deepsparse_queue import (
    DeepSparseQueue, GenerateRequest
)

class DeepSparseRouter:
    def __init__(self, service: DeepSparseService):
        self.service: DeepSparseService = service
        self.queue: DeepSparseQueue = DeepSparseQueue()
        self.cv: Condition = Condition()

    def generate(self):
        pass

    def prefill(
        self, 
        batch: Batch, 
        generation_requests: Dict[int,GenerateRequest]
    ) -> Optional[CachedBatch]:
        
        generation, next_batch = self.service.Prefill(
            PrefillRequest(batch=batch)
        )

        self.filter_notify_update([generation], generation_requests)

        return self.filter_batch(
            batch=next_batch,
            generation_requests=generation_requests
        )

    def decode(self):
        pass
    
    def filter_notify_update(
        self, 
        generations: List[Generation],
        generation_requests: Dict[int, GenerateRequest]
    ):  
        for generation in generations:
            request_id = generation.request_id

            # if we hit a stopping criteria
            if generation.generated_text is None:
                # remove from active requests and notify
                stopped_generation_request = generation_requests.pop()
                stopped_generation_request[request_id].cv.notify()

            # otherwise, update generation
            else:
                generation_requests[request_id].generation += generation.generated_text

    def filter_batch(
        self,
        batch: CachedBatch,
        generation_requests: Dict[int, GenerateRequest]
    ) -> Optional[CachedBatch]:
        
        # no need to filter
        if len(batch) == len(generation_requests):
            return batch

        # retain only requests that are still in active generation requests   
        batch.request_ids = [id for id in batch.request_ids if id in generation_requests]

        # if all requests complete, clear cache and return None
        if len(batch) == 0:
            self.service.ClearCache()
            return None
        
        # otherwise call the filter batch service
        return self.service.FilterBatch(
            FilterBatchRequest(
                batch_id=batch.batch_id,
                request_ids=batch.request_ids, 
            )
        )

    def batching_task(self):
        while True:
            with self.cv:
                while self.queue.is_empty():
                    self.cv.wait()            
            
            # loop until the queue is empty
            next_batch = self.queue.next_batch()
            while next_batch is not None:                
                cached_batch = self.prefill(*next_batch)
                
                
                
                next_batch = self.queue.next_batch()
        