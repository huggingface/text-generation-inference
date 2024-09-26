from loguru import logger
import torch
from dataclasses import dataclass
import os
from typing import List, Optional, Type

from text_generation_server.models import CausalLM
from text_generation_server.models.causal_lm import CausalLMBatch


@dataclass
class StarCoderCausalLMBatch(CausalLMBatch):
    past_key_values: Optional[List[torch.Tensor]]

    def detach_kv_cache(self):
        past_keys = []
        past_values = []
        last_dim = int(self.past_key_values[0].size(dim=-1)/2)
        for key_value in self.past_key_values:
            past_keys.append(key_value.split((last_dim, last_dim), dim=-1)[0])
            past_values.append(key_value.split((last_dim, last_dim), dim=-1)[1])
        del self.past_key_values

        return past_keys, past_values

    def attach_kv_cache(self, past_keys, past_values):
        self.past_key_values = [
            torch.cat((key, value), dim=-1) for key, value in zip(past_keys, past_values)]


class StarCoder(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        super(StarCoder, self).__init__(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return StarCoderCausalLMBatch