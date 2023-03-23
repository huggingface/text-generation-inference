import torch

from abc import ABC, abstractmethod
from typing import List, TypeVar, Type
from transformers import PreTrainedTokenizerBase

from text_generation_server.models.types import Batch, Generation
from text_generation_server.pb import generate_pb2

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.all_special_ids = set(tokenizer.all_special_ids)
        self.device = device
        self.skip_special_tokens = skip_special_tokens

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(self, batch: B) -> List[Generation]:
        raise NotImplementedError

    @staticmethod
    def get_indices_to_keep(
            requests: List[generate_pb2.Request], completed_ids: List[int],
    ) -> List[int]:
        # Compile list of indices to retain
        next_batch_keep_indices = []
        completed = iter(completed_ids)
        next_id = next(completed)
        for i, r in enumerate(requests):
            while next_id is not None and r.id > next_id:
                next_id = next(completed, None)
            if r.id != next_id:
                next_batch_keep_indices.append(i)
        return next_batch_keep_indices
