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

    def generate(self, prompt:str) -> str:
        generate_request = GenerateRequest(
            prompt=prompt,
            max_generated_tokens=100
        )

        with self.cv:
            # print("router: acquired cv")
            self.queue.append(generate_request)
            self.cv.notify()

        if prompt == "stop":
            return "stop"

        with generate_request.cv:
            # print("generate_request: acquired cv")
            if not generate_request.is_stopped:
                # print("generate_request: waiting")
                generate_request.cv.wait()

            # print("generate_request: done waiting")

        return generate_request.generation

    def prefill(
        self, 
        batch: Batch, 
        generate_requests: Dict[int,GenerateRequest]
    ) -> Optional[CachedBatch]:
        # print("prefill")
        generation, next_batch = self.service.Prefill(
            PrefillRequest(batch=batch)
        )

        self.filter_notify_update([generation], generate_requests)

        return self.filter_batch(
            batch=next_batch,
            generate_requests=generate_requests
        )

    def decode(
        self,
        batches: List[CachedBatch],
        generate_requests: Dict[int,GenerateRequest]
    ) -> Optional[CachedBatch]:
        # print("decode")
        generations, next_batch = self.service.Decode(
            DecodeRequest(batches=batches)
        )

        self.filter_notify_update(generations, generate_requests)

        return self.filter_batch(
            batch=next_batch,
            generate_requests=generate_requests
        )
    
    def filter_notify_update(
        self, 
        generations: List[Generation],
        generate_requests: Dict[int, GenerateRequest]
    ):  
        # print("filter_notify_update")
        for generation in generations:
            request_id = generation.request_id

            # if we hit a stopping criteria
            if generation.generated_text is None:
                # remove from active requests and notify
                stopped_generate_request = generate_requests.pop(request_id)
                with stopped_generate_request.cv:
                    stopped_generate_request.is_stopped = True
                    stopped_generate_request.cv.notify()

            # otherwise, update generation
            else:
                generate_requests[request_id].generation += generation.generated_text

    def filter_batch(
        self,
        batch: Optional[CachedBatch],
        generate_requests: Dict[int, GenerateRequest]
    ) -> Optional[CachedBatch]:
        # print("filter_batch")
        
        # batch is already done
        if batch is None:
            return batch
        
        # no need to filter
        if len(batch) == len(generate_requests):
            return batch

        # retain only requests that are still in active generation requests   
        batch.request_ids = [id for id in batch.request_ids if id in generate_requests]

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

def batching_task(
    router: DeepSparseRouter
) -> bool:
    # infinite_loop
    while True:
        # block while the queue is empty
        # print("batching_task: about to acquire cv")
        with router.cv:
            while router.queue.is_empty():
                # print(f"batching_task cv: waiting")
                router.cv.wait()
            # print(f"batching_task: done waiting")

        # loop until all batches in the queue are processed
        next_batch = router.queue.next_batch()
        while next_batch is not None:
            batch, generate_requests = next_batch
            
            # hack to break out of the cycle
            if batch.requests[0].prompt == "stop":
                assert router.queue.is_empty()
                assert len(router.service.cache) == 0
                return True

            cached_batch = router.prefill(
                batch=batch, 
                generate_requests=generate_requests
            )
            
            # loop until we do not reiceve any cached batch from the service (== until
            # all requests have met their stopping criteria
            while cached_batch is not None:
                # print(f"batch_size = {len(cached_batch)}")
                batches = [cached_batch]
                
                # try to get a new batch and run prefill on this batch
                next_batch = router.queue.next_batch()
                if next_batch is not None:
                    new_batch, new_generate_requests = next_batch
                    new_cached_batch = router.prefill(
                        batch=new_batch,
                        generate_requests=new_generate_requests
                    )

                    if new_cached_batch is not None:
                        batches.append(new_cached_batch)
                        assert len(generate_requests.keys() & new_generate_requests.keys()) == 0
                        generate_requests.update(new_generate_requests)

                # run decode
                cached_batch = router.decode(
                    batches=batches,
                    generate_requests=generate_requests
                )

            next_batch = router.queue.next_batch()