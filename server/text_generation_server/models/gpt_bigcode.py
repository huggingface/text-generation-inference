import torch

from dataclasses import dataclass

from opentelemetry import trace
from transformers import AutoTokenizer
from typing import Optional, Type

from text_generation_server.models.vectorized_causal_lm import (
    VectorizedCausalLM,
    VectorizedCausalLMBatch,
)
from text_generation_server.models.custom_modeling.gpt_bigcode_modeling import (
    GPTBigCodeForCausalLM,
)


tracer = trace.get_tracer(__name__)


@dataclass
class BigcodeBatch(VectorizedCausalLMBatch):
    kv_cache_seq_dim: int = 1

    def _filter_kv_caches(self, keep_indices, sequence_slice):
        if self.past_key_values is not None:
            for layer_kv, _ in self.past_key_values:
                # Update tensors in-place to allow incremental garbage collection
                layer_kv.data = layer_kv[keep_indices, sequence_slice]

    @classmethod
    def _concatenate_key_values(
        cls, batches, start_indices, end_indices, left_indices, max_input_length
    ):
        device = batches[0].input_ids.device
        batch_size = sum([len(batch.requests) for batch in batches])

        for batch in batches:
            if batch.past_key_values is None:
                raise ValueError("Only concatenate prefilled batches")

        past_key_values = []
        for kv_caches in zip(*(batch.past_key_values for batch in batches)):
            key_values, seq_lengths = zip(*kv_caches)
            assert all(
                left_index + seq_length == max_input_length
                for left_index, seq_length in zip(left_indices, seq_lengths)
            )

            allocate_seq_len = max(
                left_index + key_value.size(1)
                for left_index, key_value in zip(left_indices, key_values)
            )
            allocate_seq_len += -allocate_seq_len % 8

            kv_cache = torch.empty(
                (batch_size, allocate_seq_len, *key_values[0].shape[2:]),
                dtype=key_values[0].dtype,
                device=device,
            )
            for key_value, start_index, end_index, left_index in zip(
                key_values,
                start_indices,
                end_indices,
                left_indices,
            ):
                kv_cache[start_index:end_index, left_index:max_input_length].copy_(
                    key_value
                )
                # Set padding to zero to avoid propagating nans.
                kv_cache[start_index:end_index, :left_index].fill_(0)
                kv_cache[start_index:end_index, max_input_length:].fill_(0)
            past_key_values.append((kv_cache, max_input_length))

    def __len__(self):
        return len(self.requests)


class BigcodeCausalLM(VectorizedCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )
        model = GPTBigCodeForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize == "bitsandbytes",
        )
        tokenizer.pad_token_id = (
            model.config.pad_token_id
            if model.config.pad_token_id is not None
            else model.config.eos_token_id
        )

        super(VectorizedCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[BigcodeBatch]:
        return BigcodeBatch

    def forward(self, batch: BigcodeBatch):
        key_length = batch.max_input_length
        query_length = key_length if batch.past_key_values is None else 1
        input_ids = batch.input_ids[:, key_length - query_length : key_length]
        # Model Forward
        logits, batch.past_key_values = self.model.forward(
            input_ids=input_ids,
            attention_mask=batch.attention_mask[:, :key_length],
            position_ids=batch.position_ids[:, key_length - query_length : key_length],
            past_key_values=batch.past_key_values,
            predict_all_tokens=batch.details,
        )
        next_token_ids, logprobs = batch.next_token_chooser(
            input_ids, logits, batch.details
        )
        # Update batch
        # TODO: Why do we need all input ids?
        batch.input_ids[:, key_length].copy_(next_token_ids)
        batch.input_lengths = [length + 1 for length in batch.input_lengths]
        batch.max_input_length += 1

        return next_token_ids, logprobs

    def mock_kv_cache(self, batch: BigcodeBatch, dtype: Optional[torch.dtype]):
        allocate_length = batch.max_input_length + -batch.max_input_length % 8
        return [
            (
                torch.empty(
                    [
                        len(batch),
                        allocate_length - 1,
                        2 * self.model.config.n_embd // self.model.config.n_head,
                    ],
                    dtype=dtype,
                    device=batch.input_ids.device,
                ),
                batch.max_input_length - 1,
            )
            for _ in range(self.model.config.n_layer)
        ]
