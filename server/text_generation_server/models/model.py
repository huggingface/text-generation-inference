import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase

from text_generation_server.models.types import Batch, GeneratedText

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.tokenizer = tokenizer
        self.all_special_ids = set(tokenizer.all_special_ids)
        self.device = device

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(self, batch: B) -> Tuple[List[GeneratedText], Optional[B]]:
        raise NotImplementedError

    def decode_token(self, previous_token_id: int, token_id: int) -> str:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        # Decode previous token and previous token + token
        results = self.tokenizer.batch_decode(
            [[previous_token_id], [previous_token_id, token_id]],
            skip_special_tokens=False,
        )
        # slice to remove previous token
        return results[1][len(results[0]) :]
