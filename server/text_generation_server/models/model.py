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

    def decode_token(
        self, all_input_ids: List[int], offset: Optional[int] = None
    ) -> Tuple[str, Optional[int]]:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""

        # Decode all token minus last one and all tokens
        results = self.tokenizer.batch_decode(
            [all_input_ids[:-1], all_input_ids],
            skip_special_tokens=False,
        )

        # default offset is only the last token
        if offset is None:
            offset = len(results[0])

        # get text
        text = results[1][offset:]

        # if text is utf-8
        if text and text[-1] != "�":
            return text, None
        else:
            return "", offset
