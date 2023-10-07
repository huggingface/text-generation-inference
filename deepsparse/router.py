from queue import Queue
from typing import List, Dict, Optional, Tuple
from service.service import DeepSparseService
from service.causal_lm import DeepSparseCausalLM
from utils import CachedBatch, Batch, Generation, GenerateRequest, Request, GenerationParameters

class DeepSparseRouter:
    def __init__(
        self, 
        service: Optional[DeepSparseService] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ):        
        assert service is not None or (model_path is not None and tokenizer_path is not None)

        if service is not None:
            self.service = service
        else:
            self.service = DeepSparseService(
                model = DeepSparseCausalLM(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path
                )
            )

        self.queue: DeepSparseQueue = DeepSparseQueue()
        self.batching_task_should_stop:bool = False

    def submit_request(self, generate_request: GenerateRequest):
        self.queue.append(generate_request)

    def stop_batching_task(self):
        # tell batching task to stop
        self.batching_task_should_stop = True
        
        # unblock the batching task with a dummy request if blocked
        self.queue.append(GenerateRequest(
            inputs="stop",
            generation_parameters=GenerationParameters(max_new_tokens=1),
            response_stream=Queue()
        ))

    def prefill(
        self,
        batch: Batch,
        generate_requests: Dict[int, GenerateRequest]
    ) -> Optional[CachedBatch]:
        
        generation, next_batch = self.service.Prefill(batch=batch)
        active_generate_request_ids = self.filter_send_generations([generation], generate_requests)
        return self.filter_batch(batch=next_batch, active_generate_request_ids=active_generate_request_ids)

    def decode(
        self,
        batches: List[CachedBatch],
        generate_requests: Dict[int,GenerateRequest]
    ) -> Optional[CachedBatch]:

        generations, next_batch = self.service.Decode(batches=batches)
        active_generate_request_ids = self.filter_send_generations(generations, generate_requests)
        return self.filter_batch(batch=next_batch, active_generate_request_ids=active_generate_request_ids)
    
    def filter_send_generations(
        self, 
        generations: List[Generation],
        generate_requests: Dict[int, GenerateRequest]
    ) -> List[int]:
        
        active_request_ids = []
        for generation in generations:
            # send generation to the response stream
            generate_requests[generation.request_id].response_stream.put(generation)

            # remove request from active requests if stopped
            if generation.stopped:
                generate_requests.pop(generation.request_id)
            else:
                active_request_ids.append(generation.request_id)
        
        return active_request_ids

    def filter_batch(
        self,
        batch: Optional[CachedBatch],
        active_generate_request_ids: List[int]
    ) -> Optional[CachedBatch]:
        
        # if batch done OR nothing to filter
        if batch is None or len(batch) == len(active_generate_request_ids):
            return batch

        # active request_ids
        batch.request_ids = active_generate_request_ids

        # if all requests complete, clear cache
        if len(batch) == 0:
            self.service.ClearCache()
            return None

        return self.service.FilterBatch(batch_id=batch.batch_id, request_ids=batch.request_ids)


# TODO: update to do more sophisticated logic as to when to do a prefill
def batching_task(router: DeepSparseRouter):
    # while not signaled to stop
    while not router.batching_task_should_stop:
        
        # loop until no requests to process (note: this blocks if queue is empty)
        next_batch = router.queue.next_batch(block=True)
        while next_batch is not None:
            batch, generate_requests = next_batch
            
            # run prefill
            cached_batch = router.prefill(
                batch=batch, 
                generate_requests=generate_requests
            )
            
            # loop until we do not reiceve any cached batch from the service 
            #   == until all active requests have met their stopping criteria
            while cached_batch is not None:
                batches = [cached_batch]
                
                # try to get a new batch and run prefill on this batch
                next_batch = router.queue.next_batch(block=False)
                
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

            next_batch = router.queue.next_batch(block=False)

# TODO: implement logic for maximum size of the queue based on memory usage
class DeepSparseQueue:
    def __init__(self):
        self.next_request_id: int = 0
        self.next_batch_id: int = 0
        self.queue: Queue[GenerateRequest] = Queue()

    def append(self, generate_request: GenerateRequest):
        self.queue.put(generate_request)

    # TODO: enable multiple prefill requests in a batch
    def next_batch(self, block=False) -> Optional[Tuple[Batch, Dict[int, GenerateRequest]]]:
                
        # if not blocking, return none if empty
        if not block and self.queue.empty():
            return None
        
        # if block = True, this blocks until something ready
        # if block = False, the queue has data (if not an exception is raised)
        #       while queue.empty() == False typically not guarentee data on next queue.get(), this 
        #       queue is only subscribed to by one thread (this one) since batching_task is the only
        #       so it does in our case

        generate_request = self.queue.get(block=block)
        generate_requests = {self.next_request_id: generate_request}

        # format into request
        request = Request(
            id=self.next_request_id,
            inputs=generate_request.inputs,
            generation_parameters=generate_request.generation_parameters,
        )
        self.next_request_id += 1
        
        # format into batch
        batch = Batch(
            id = self.next_batch_id,
            requests=[request]
        )
        self.next_batch_id += 1

        # return batch, generate_requests
        return (batch, generate_requests)
