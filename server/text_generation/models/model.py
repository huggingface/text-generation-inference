import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase

from text_generation.models.types import Batch, GeneratedText

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(self, batch: B) -> Tuple[List[GeneratedText], Optional[B]]:
        raise NotImplementedError
