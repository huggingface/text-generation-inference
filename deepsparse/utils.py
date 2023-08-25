from dataclasses import dataclass
from queue import Queue
from enum import Enum
from typing import List, Optional

class FinishReason(Enum):
    FINISH_REASON_LENGTH = 1
    FINISH_REASON_EOS_TOKEN = 2

class StoppingCriteria:
    def __init__(
        self, 
        eos_token_id: int,
        max_new_tokens: int,
    ):        
        assert max_new_tokens > 0
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.current_tokens = 0

    def __call__(self, generated_token_id:int):
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True, FinishReason.FINISH_REASON_LENGTH
        
        if generated_token_id == self.eos_token_id:
            return True, FinishReason.FINISH_REASON_EOS_TOKEN
        
        return False, None

@dataclass
class Request:
    id: int
    inputs: str
    max_new_tokens: int

@dataclass
class Batch:
    id: int
    requests: List[Request]

@dataclass
class CachedBatch:
    batch_id: int
    request_ids: List[int]

    def __len__(self):
        return len(self.request_ids)

@dataclass
class Generation:
    request_id: int
    token: Optional[str]
    token_id: Optional[str]
    stopped: bool
    finish_reason: FinishReason = None

@dataclass
class GenerateRequest:
    inputs: str
    max_new_tokens: int
    response_stream: Queue[Generation]