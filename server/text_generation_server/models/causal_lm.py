from operator import itemgetter

import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser

tracer = trace.get_tracer(__name__)


@dataclass
class CausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # All tokens
    all_input_ids: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]

    # Metadata used for padding
    size: int
    max_input_length: int
    padding_right_offset: int

    # Past metadata
    keys_head_dim_last: bool = True

    def to_pb(self) -> generate_pb2.Batch:
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=self.size,
        )

    def get_id(self) -> int:
        return self.batch_id

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "CausalLMBatch":
        inputs = []
        next_token_choosers = []

        # Parse batch
        padding_right_offset = 0
        for r in pb.requests:
            inputs.append(r.inputs)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))
            padding_right_offset = max(padding_right_offset, r.max_new_tokens)

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(device)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()

        input_ids = tokenized_inputs["input_ids"]
        # Allocate maximum attention_mask
        attention_mask = input_ids.new_zeros(
            (pb.size, max_input_length + padding_right_offset)
        )
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :max_input_length] = tokenized_inputs["attention_mask"]

        position_ids = tokenized_inputs["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(tokenized_inputs["attention_mask"] == 0, 1)
        all_input_ids = input_ids.unsqueeze(-1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths.tolist(),
            next_token_choosers=next_token_choosers,
            size=pb.size,
            max_input_length=max_input_length.item(),
            padding_right_offset=padding_right_offset,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["CausalLMBatch"]) -> "CausalLMBatch":
        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += batch.size
            max_input_length = max(max_input_length, batch.max_input_length)
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        input_lengths = []
        all_input_ids = []
        next_token_choosers = []

        # Batch tensors
        input_ids = None
        attention_mask = None
        position_ids = None
        past_key_values = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)

            # Slicing end index for this batch
            end_index = start_index + batch.size

            # We only concatenate batches that did at least one step
            if batch.past_key_values is None:
                raise ValueError("only concatenate prefilled batches")

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((total_batch_size, 1))
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_input_length + padding_right_offset),
                )

            # We need to slice the attention mask to remove padding from previous steps
            # and to remove unused allocated space
            left_offset = max_input_length - batch.max_input_length
            batch_left_offset = (
                batch.attention_mask.shape[1]
                - batch.max_input_length
                - batch.padding_right_offset
            )
            attention_mask[
                start_index:end_index,
                left_offset:-padding_right_offset,
            ] = batch.attention_mask[
                :,
                batch_left_offset : -batch.padding_right_offset,
            ]

            # Create empty tensor
            # position_ids is always of shape [batch_size, 1]
            if position_ids is None:
                position_ids = batch.position_ids.new_empty((total_batch_size, 1))
            position_ids[start_index:end_index] = batch.position_ids

            for j, past in enumerate(batch.past_key_values):
                past_keys, past_values = past

                # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
                # BLOOM Keys:   [batch_size * num_heads, head_dim, seq_length]
                # BLOOM Values: [batch_size * num_heads, seq_length, head_dim]
                past_keys = past_keys.view(batch.size, -1, *past_keys.shape[-2:])
                past_values = past_values.view(batch.size, -1, *past_values.shape[-2:])

                _, num_heads, padded_sequence_length, head_dim = past_values.shape

                padded_past_values_shape = (
                    total_batch_size,
                    num_heads,
                    max_input_length - 1,
                    head_dim,
                )

                if batch.keys_head_dim_last:
                    padded_past_keys_shape = padded_past_values_shape
                else:
                    # seq_length is last for BLOOM
                    padded_past_keys_shape = (
                        total_batch_size,
                        num_heads,
                        head_dim,
                        max_input_length - 1,
                    )

                # This will run only once per layer
                if j == len(past_key_values):
                    padded_past_keys = past_keys.new_zeros(padded_past_keys_shape)
                    padded_past_values = past_values.new_zeros(padded_past_values_shape)
                    past_key_values.append((padded_past_keys, padded_past_values))

                # We slice the past keys and values to remove the padding from previous batches
                if batch.keys_head_dim_last:
                    past_key_values[j][0][
                        start_index:end_index,
                        :,
                        -(batch.max_input_length - 1) :,
                        :,
                    ] = past_keys[:, :, -(batch.max_input_length - 1) :, :]
                else:
                    past_key_values[j][0][
                        start_index:end_index,
                        :,
                        :,
                        -(batch.max_input_length - 1) :,
                    ] = past_keys[:, :, :, -(batch.max_input_length - 1) :]

                past_key_values[j][1][
                    start_index:end_index, :, -(batch.max_input_length - 1) :, :
                ] = past_values[:, :, -(batch.max_input_length - 1) :, :]

            start_index += batch.size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            next_token_choosers=next_token_choosers,
            size=total_batch_size,
            max_input_length=max_input_length,
            padding_right_offset=padding_right_offset,
            keys_head_dim_last=batches[0].keys_head_dim_last,
        )

    def __len__(self):
        return len(self.requests)

    @classmethod
    def prune(cls, batch: "CausalLMBatch", completed_ids: List[int]) -> Optional["CausalLMBatch"]:
        """Prune completed entries from a batch"""

        if not completed_ids:
            # Nothing to prune
            return batch

        # Compile list of indices to retain
        keep_indices = Model.get_indices_to_keep(batch.requests, completed_ids)
        new_size = len(keep_indices)

        # If the whole batch has finished, discard it
        if new_size == 0:
            return None

        #TODO maybe a single loop for all these list slices
        slice_list = itemgetter(*keep_indices) if new_size > 1 else lambda l: (l[keep_indices[0]],)
        batch.input_lengths = list(slice_list(batch.input_lengths))
        batch.requests = slice_list(batch.requests)
        batch.all_input_ids = slice_list(batch.all_input_ids)
        batch.next_token_choosers = slice_list(batch.next_token_choosers)

        batch.max_input_length = max(batch.input_lengths)

        # Force past to be of dim [batch_size, num_heads, ...] for easy indexing
        batch.past_key_values = [
            [t.view(batch.size, -1, *t.shape[-2:])[keep_indices] for t in layer]
            for layer in batch.past_key_values
        ]
        batch.input_ids = batch.input_ids[keep_indices]
        batch.attention_mask = batch.attention_mask[keep_indices]
        batch.position_ids = batch.position_ids[keep_indices]

        batch.size = new_size

        return batch


class CausalLM(Model):
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False, skip_special_tokens=True):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize,
        ).eval()
        tokenizer.pad_token_id = (
            self.model.config.pad_token_id
            if self.model.config.pad_token_id is not None
            else self.model.config.eos_token_id
        )

        super(CausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return CausalLMBatch

    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    @tracer.start_as_current_span("generate_token")
    def generate_token(self, batch: CausalLMBatch, prefill: bool = False) -> List[Generation]:
        # slice the attention mask to the correct shape
        attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]

        logits, past = self.forward(
            batch.input_ids,
            attention_mask,
            batch.position_ids,
            batch.past_key_values,
        )

        # New values for next forward
        next_batch_input_ids = []
        next_batch_all_input_ids = []

        # Results
        generations: List[Generation] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            logits,
            batch.next_token_choosers,
            batch.all_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            logits,
            next_token_chooser,
            all_input_ids,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits
            )

            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id])
            new_input_length = input_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()

            next_batch_input_ids.append(next_token_id)
            next_batch_all_input_ids.append(all_input_ids)
            batch.input_lengths[i] = new_input_length

            # Prefill
            if prefill:
                # Remove generated token to only have prefill and add nan for first prompt token
                prefill_logprobs = [float("nan")] + logprobs.gather(
                    1, all_input_ids[1:]
                ).squeeze(1)[-new_input_length:-1].tolist()
                prefill_token_ids = all_input_ids[-new_input_length:-1]
                prefill_tokens = PrefillTokens(prefill_token_ids, prefill_logprobs)
            else:
                prefill_tokens = None

            generation = Generation(
                request.id,
                prefill_tokens,
                next_token_id_squeezed,
                next_token_logprob,
                next_token_id_squeezed.item() in self.all_special_ids,
            )

            generations.append(generation)

        # Update attention_mask as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1

        # Update position_ids
        batch.position_ids = batch.position_ids[:, -1:] + 1

        batch.input_ids = torch.cat(next_batch_input_ids, dim=0)
        batch.past_key_values = past
        batch.all_input_ids = next_batch_all_input_ids
        batch.max_input_length += 1
        batch.padding_right_offset -= 1

        return generations
