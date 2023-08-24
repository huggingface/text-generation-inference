from dataclasses import dataclass
from queue import Queue
from typing import List, Optional

@dataclass
class Request:
    id: int
    prompt: str
    max_generated_tokens: int

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

@dataclass
class GenerateRequest:
    prompt: str
    max_generated_tokens: int
    response_stream: Queue[Generation]