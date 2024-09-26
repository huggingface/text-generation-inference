# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import torch

from typing import Optional, Type

from transformers import PreTrainedTokenizerBase

from text_generation_server.models import CausalLM
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.pb import generate_pb2


class BloomCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "CausalLMBatch":
        batch = super().from_pb(
            pb=pb,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
        )
        batch.keys_head_dim_last = False
        return batch


class BLOOM(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        super(BLOOM, self).__init__(
            model_id=model_id,
            revision=revision,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return BloomCausalLMBatch
