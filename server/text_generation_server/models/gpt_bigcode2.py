import torch

from dataclasses import dataclass

from opentelemetry import trace
from transformers import AutoTokenizer
from typing import Optional, Type, List
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

from text_generation_server.pb import generate_pb2
from text_generation_server.models.vectorized_causal_lm import (
    VectorizedCausalLM,
    VectorizedCausalLMBatch,
)
from text_generation_server.models.custom_modeling.gpt_bigcode2_modeling import (
    GPTBigCodeForCausalLM as GPTBigCode2ForCausalLM,
)
from text_generation_server.models.custom_modeling.gpt_bigcode3_modeling import (
    GPTBigCodeForCausalLM as GPTBigCode3ForCausalLM,
)
from text_generation_server.models.custom_modeling.gpt_bigcode4_modeling import (
    GPTBigCodeForCausalLM as GPTBigCode4ForCausalLM,
)
from transformers.modeling_utils import PreTrainedModel

tracer = trace.get_tracer(__name__)


@dataclass
class Bigcode2Batch(VectorizedCausalLMBatch):
    kv_cache_seq_dim: int = 1
    pad_key_length_to_multiple: int = 8

    # Prefill the attention mask for padded key length.
    attention_mask_fill_value = False

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "Bigcode2Batch":
        batch = super().from_pb(pb, tokenizer, device)
        batch.attention_mask[:, batch.max_input_length :].fill_(False)
        return batch

    def _filter_kv_caches(self, keep_indices, sequence_slice):
        if self.past_key_values is not None:
            for layer_kv in self.past_key_values:
                # Update tensors in-place to allow incremental garbage collection
                layer_kv.data = layer_kv[keep_indices, sequence_slice]

    @classmethod
    def concatenate(cls, batches: List["Bigcode2Batch"]) -> "Bigcode2Batch":
        batch = super().concatenate(batches)
        # Replace the attention mask with zeros to support padded key length.
        # They are already filled with ones in super, but duplication is needed to generate the position ids.
        batch.attention_mask[:, batch.max_input_length :].fill_(False)
        return batch

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
        for key_values in zip(*(batch.past_key_values for batch in batches)):
            allocate_seq_len = max(
                left_index + key_value.size(1)
                for left_index, key_value in zip(left_indices, key_values)
            )
            allocate_seq_len += (
                -allocate_seq_len % batches[0].pad_key_length_to_multiple
            )

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
            past_key_values.append(kv_cache)

    def __len__(self):
        return len(self.requests)


class Bigcode2CausalLMBase(VectorizedCausalLM):
    #model: GPTBigCode2ForCausalLM
    _model_class:Type[PreTrainedModel]

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
        model = self._model_class.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize == "bitsandbytes",
        )
        model.post_load_weights()

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
    def batch_type(self) -> Type[Bigcode2Batch]:
        return Bigcode2Batch

    def forward(self, batch: Bigcode2Batch):
        key_length = batch.max_input_length
        if batch.past_key_values is None:
            # Prefill (flash attn, unpadded key length)
            input_ids = batch.input_ids[:, :key_length]
            logits, batch.past_key_values = self.model.prefill(
                input_ids=input_ids,
                attention_mask=batch.attention_mask[:, :key_length],
                position_ids=batch.position_ids[:, :key_length],
                predict_all_tokens=batch.details,
            )
        else:
            # Decode (fused attn, padded key length)
            batch.attention_mask[:, key_length - 1].fill_(True)
            padded_key_length = (
                key_length + -key_length % batch.pad_key_length_to_multiple
            )
            input_ids = batch.input_ids[:, key_length - 1]
            # Model Forward
            logits, batch.past_key_values = self.model.decode(
                input_ids=input_ids,
                attention_mask=batch.attention_mask[:, None, :padded_key_length],
                position_ids=batch.position_ids[:, key_length - 1],
                past_key_values=batch.past_key_values,
                key_length=key_length,
            )

        next_token_ids, logprobs = batch.next_token_chooser(
            input_ids.unsqueeze(1), logits.unsqueeze(1), batch.details
        )
        # Update batch
        # TODO: Why do we need all input ids?
        batch.input_ids[:, key_length].copy_(next_token_ids)
        batch.input_lengths = [length + 1 for length in batch.input_lengths]
        batch.max_input_length += 1

        return next_token_ids, logprobs

    def mock_kv_cache(self, batch: Bigcode2Batch, dtype: Optional[torch.dtype]):
        allocate_length = (
            batch.max_input_length
            + -batch.max_input_length % batch.pad_key_length_to_multiple
        )
        return [
            torch.randn(
                [
                    len(batch),
                    allocate_length - 1,
                    2 * self.model.config.n_embd // self.model.config.n_head,
                ],
                dtype=dtype,
                device=batch.input_ids.device,
            )
            for _ in range(self.model.config.n_layer)
        ]

class Bigcode2CausalLM(Bigcode2CausalLMBase):
    _model_class=GPTBigCode2ForCausalLM


class Bigcode3CausalLM(Bigcode2CausalLMBase):
    _model_class=GPTBigCode3ForCausalLM


class Bigcode4CausalLM(Bigcode2CausalLMBase):
    _model_class = GPTBigCode4ForCausalLM
