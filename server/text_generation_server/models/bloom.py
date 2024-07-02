import torch
import torch.distributed

from typing import Optional, Type

from transformers import (
    PreTrainedTokenizerBase,
)

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
        batch = super().from_pb(pb=pb, tokenizer=tokenizer, dtype=dtype, device=device)
        batch.keys_head_dim_last = False
        return batch


class BLOOMSharded(CausalLM):
    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return BloomCausalLMBatch

    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ):
        outputs, speculative_logits = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits
        return logits, speculative_logits, outputs.past_key_values
