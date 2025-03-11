import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

from text_generation_server.pb import generate_pb2
from text_generation_server.pb.generate_pb2 import FinishReason


class Batch(ABC):
    @abstractmethod
    def to_pb(self) -> generate_pb2.CachedBatch:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def filter(self, request_ids: List[int]) -> "Batch":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


@dataclass
class GeneratedText:
    text: str
    generated_tokens: int
    finish_reason: FinishReason
    seed: Optional[int]

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(
            text=self.text,
            generated_tokens=self.generated_tokens,
            finish_reason=self.finish_reason,
            seed=self.seed,
        )


@dataclass
class Tokens:
    token_ids: List[int]
    logprobs: List[float]
    texts: List[str]
    is_special: List[bool]

    def to_pb(self) -> generate_pb2.Tokens:
        return generate_pb2.Tokens(
            ids=self.token_ids,
            logprobs=self.logprobs,
            texts=self.texts,
            is_special=self.is_special,
        )

    def __len__(self):
        return len(self.token_ids)


@dataclass
class Generation:
    request_id: int
    prefill_tokens: Optional[Tokens]
    tokens: Tokens
    generated_text: Optional[GeneratedText]
    # Optional for now, since it's not yet supported for every model.
    top_tokens: Optional[List[Tokens]]

    def to_pb(self) -> generate_pb2.Generation:
        return generate_pb2.Generation(
            request_id=self.request_id,
            prefill_tokens=(
                self.prefill_tokens.to_pb() if self.prefill_tokens is not None else None
            ),
            tokens=self.tokens.to_pb(),
            generated_text=(
                self.generated_text.to_pb() if self.generated_text is not None else None
            ),
            top_tokens=(
                [top_tokens.to_pb() for top_tokens in self.top_tokens]
                if self.top_tokens is not None
                else None
            ),
        )
