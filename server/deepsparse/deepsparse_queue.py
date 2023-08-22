from typing import Deque, Optional, Tuple, Dict
from collections import deque
from threading import Condition
from server.deepsparse.deepsparse_requests import Batch, Request

class GenerateRequest:    
    def __init__(
        self,
        prompt: str,
        max_generated_tokens: int
    ):
        self.prompt = prompt
        self.generation = prompt
        self.max_generated_tokens = max_generated_tokens
        self.cv = Condition()

class DeepSparseQueue:
    def __init__(self):
        self.next_request_id: int = 0
        self.next_batch_id: int = 0
        self.queue: Deque[GenerateRequest] = deque()

    def append(self, generate_request: GenerateRequest):
        self.queue.append(generate_request)
    
    def is_empty(self):
        return len(self.queue) == 0

    # (todo): enable multiple prefill requests in a batch
    def next_batch(self) -> Optional[Tuple[Batch, Dict[int, GenerateRequest]]]:
        if self.is_empty():
            return None

        # pop first generate_request in the queue
        generate_request = self.queue.popleft()
        generate_requests = {
            self.next_request_id: generate_request
        }

        # format into request
        request = Request(
            id=self.next_request_id,
            prompt=generate_request.prompt,
            max_generated_tokens=generate_request.max_generated_tokens
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