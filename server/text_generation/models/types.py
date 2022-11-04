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
    output: str
    tokens: int

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(
            request=self.request, output=self.output, tokens=self.tokens
        )
