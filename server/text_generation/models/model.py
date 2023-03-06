import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase

from text_generation.models.types import Batch, GeneratedText

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.tokenizer = tokenizer
        self.all_special_ids = set(tokenizer.all_special_ids)
        self.device = device

        # see `decode_token` method
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<decode-token>"]}
        )
        self.special_decode_token_id = self.tokenizer.convert_tokens_to_ids(
            "<decode-token>"
        )
        self.special_decode_token_length = len("<decode-token>")

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(self, batch: B) -> Tuple[List[GeneratedText], Optional[B]]:
        raise NotImplementedError

    def decode_token(self, token_id: int) -> str:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        # append token to special decode token and decode both
        result = self.tokenizer.decode(
            [self.special_decode_token_id, token_id], skip_special_tokens=False
        )
        # slice to remove special decode token
        return result[self.special_decode_token_length :]
