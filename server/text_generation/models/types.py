import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer

from text_generation.pb import generate_pb2


class Batch(ABC):
    @abstractmethod
    def to_pb(self) -> generate_pb2.Batch:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pb(
        cls, pb: generate_pb2.Batch, tokenizer: AutoTokenizer, device: torch.device
    ) -> "Batch":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        raise NotImplementedError


@dataclass
class GeneratedText:
    request: generate_pb2.Request
    output_text: str
    generated_tokens: int
    tokens: List[str]
    token_ids: List[int]
    logprobs: List[float]
    reason: str

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(
            request=self.request,
            output_text=self.output_text,
            generated_tokens=self.generated_tokens,
            tokens=self.tokens,
            token_ids=self.token_ids,
            logprobs=self.logprobs,
            finish_reason=self.reason,
        )
