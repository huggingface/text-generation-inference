from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Request:
    id: int
    prompt: str

@dataclass
class Batch:
    id: int
    requests: List[Request]

@dataclass
class CachedBatch:
    batch_id: int
    request_ids: List[int]

@dataclass
class Generation:
    request_id: int
    generated_text: Optional[str]

@dataclass  
class PrefillRequest:
    batch: Batch

@dataclass
class DecodeRequest:
    batches: List[CachedBatch]

@dataclass
class FilterBatchRequest:
    batch_id: int
    request_ids: List[int]