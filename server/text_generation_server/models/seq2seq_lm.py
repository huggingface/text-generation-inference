from operator import itemgetter

import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    Generation,
    PrefillTokens,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser

tracer = trace.get_tracer(__name__)


@dataclass
class Seq2SeqLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Encoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    # Decoder values
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: Optional[torch.Tensor]
    encoder_last_hidden_state: Optional[torch.Tensor]

    # Seq2SeqLM keeps track of both encoder and decoder attention keys and values
    past_key_values: Optional[List[Tuple]]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    decoder_input_lengths: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]

    # Metadata used for padding
    size: int
    max_input_length: int
    max_decoder_input_length: int
    padding_right_offset: int

    def to_pb(self) -> generate_pb2.Batch:
        """Convert a Seq2SeqLMBatch to a text_generation_server.v1.Batch protobuf"""
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
    ) -> "Seq2SeqLMBatch":
        """Convert a text_generation_server.v1.Batch protobuf to a Seq2SeqLMBatch"""
        inputs = []
        next_token_choosers = []

        decoder_input_ids = []
        decoder_input_lengths = []

        # Parse batch
        padding_right_offset = 0
        for r in pb.requests:
            inputs.append(r.inputs)
            # Decoder sequence only contains the bos_token
            decoder_input_ids.append(tokenizer.bos_token_id)
            decoder_input_lengths.append(1)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))
            padding_right_offset = max(
                padding_right_offset, r.max_new_tokens
            )

        # Tokenize batch
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(device)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()

        # Convert decoder_input_ids to torch tensor of size [batch_size, 1]
        decoder_input_ids = torch.tensor(decoder_input_ids, device=device).unsqueeze(-1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=None,
            encoder_last_hidden_state=None,
            past_key_values=None,
            input_lengths=input_lengths.tolist(),
            decoder_input_lengths=decoder_input_lengths,
            next_token_choosers=next_token_choosers,
            size=len(pb.requests),
            max_input_length=max_input_length.item(),
            max_decoder_input_length=1,
            padding_right_offset=padding_right_offset,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["Seq2SeqLMBatch"]) -> "Seq2SeqLMBatch":
        """Concatenate multiple batches together by padding internal torch tensors"""

        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        max_decoder_input_length = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += batch.size
            max_input_length = max(max_input_length, batch.max_input_length)
            max_decoder_input_length = max(
                max_decoder_input_length, batch.max_decoder_input_length
            )
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        input_lengths = []
        decoder_input_lengths = []
        next_token_choosers = []

        # Batch tensors
        attention_mask = None
        decoder_input_ids = None
        decoder_attention_mask = None
        encoder_last_hidden_state = None
        past_key_values = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0

        for i, batch in enumerate(batches):
            # Extend all list attributes
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            decoder_input_lengths.extend(batch.decoder_input_lengths)
            next_token_choosers.extend(batch.next_token_choosers)

            # Slicing end index for this batch
            end_index = start_index + batch.size

            # We only concatenate batches that did at least one step
            if batch.encoder_last_hidden_state is None:
                raise ValueError("Batch encoder_last_hidden_state cannot be None")

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_input_length),
                )
            # Copy to correct indices
            attention_mask[
                start_index:end_index, -batch.max_input_length :
            ] = batch.attention_mask[:, -batch.max_input_length :]

            # Create padded tensor
            if decoder_input_ids is None:
                decoder_input_ids = batch.decoder_input_ids.new_zeros(
                    (total_batch_size, max_decoder_input_length),
                )
            # Copy to correct indices
            decoder_input_ids[
                start_index:end_index, -batch.max_decoder_input_length :
            ] = batch.decoder_input_ids[:, -batch.max_decoder_input_length :]

            # Create padded tensor
            if decoder_attention_mask is None:
                # As decoder_attention_mask might not exist, we use `batch.attention_mask` for device here
                decoder_attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_decoder_input_length + padding_right_offset),
                )
            # If the decoder mask does not exist yet, all generations started at the same time and we never concatenated
            # this batch. All generations are of length `batch.max_decoder_input_length`.
            left_offset = max_decoder_input_length - batch.max_decoder_input_length
            if batch.decoder_attention_mask is None:
                decoder_attention_mask[
                    start_index:end_index,
                    left_offset:-padding_right_offset,
                ] = 1
            # If it exists, we need to index
            else:
                batch_left_offset = (
                    batch.decoder_attention_mask.shape[1]
                    - batch.max_decoder_input_length
                    - batch.padding_right_offset
                )
                decoder_attention_mask[
                    start_index:end_index,
                    left_offset:-padding_right_offset,
                ] = batch.decoder_attention_mask[
                    :,
                    batch_left_offset : -batch.padding_right_offset,
                ]

            # Create padded tensor
            if encoder_last_hidden_state is None:
                encoder_last_hidden_state = batch.encoder_last_hidden_state.new_zeros(
                    (
                        total_batch_size,
                        max_input_length,
                        batch.encoder_last_hidden_state.shape[-1],
                    ),
                )

            # Copy to correct indices
            encoder_last_hidden_state[
                start_index:end_index, -batch.max_input_length :, :
            ] = batch.encoder_last_hidden_state[:, -batch.max_input_length :, :]

            # Iterate over attention layers
            for j, past in enumerate(batch.past_key_values):
                _, num_heads, _, head_dim = past[0].shape

                # This will run only once per layer
                if j == len(past_key_values):
                    past_key_values.append([])

                # Decoder past
                for k, t in enumerate(past[:2]):
                    padded_t_shape = (
                        total_batch_size,
                        num_heads,
                        (max_decoder_input_length - 1),
                        head_dim,
                    )

                    # Initialize tensors
                    # This will run only once per layer and per past tensor
                    if k == len(past_key_values[j]):
                        past_key_values[j].append(t.new_zeros(padded_t_shape))

                    # We slice the past keys and values to remove the padding from previous batches
                    past_key_values[j][k][
                        start_index:end_index,
                        :,
                        -(batch.max_decoder_input_length - 1) :,
                        :,
                    ] = t[:, :, -(batch.max_decoder_input_length - 1) :, :]

                # encoder past
                for k, t in enumerate(past[2:]):
                    padded_t_shape = (
                        total_batch_size,
                        num_heads,
                        max_input_length,
                        head_dim,
                    )

                    idx = k + 2

                    # Initialize tensors
                    # This will run only once per layer and per past tensor
                    if idx == len(past_key_values[j]):
                        past_key_values[j].append(t.new_zeros(padded_t_shape))

                    past_key_values[j][idx][
                        start_index:end_index, :, -batch.max_input_length :, :
                    ] = t[:, :, -batch.max_input_length :, :]

            start_index += batch.size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_last_hidden_state=encoder_last_hidden_state,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            next_token_choosers=next_token_choosers,
            size=total_batch_size,
            max_input_length=max_input_length,
            max_decoder_input_length=max_decoder_input_length,
            padding_right_offset=padding_right_offset,
        )

    def __len__(self):
        return len(self.requests)

    @classmethod
    def prune(cls, batch: "Seq2SeqLMBatch", completed_ids: List[int]) -> Optional["Seq2SeqLMBatch"]:
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
        batch.input_lengths = slice_list(batch.input_lengths)
        batch.decoder_input_lengths = slice_list(batch.decoder_input_lengths)
        batch.requests = slice_list(batch.requests)
        batch.next_token_choosers = slice_list(batch.next_token_choosers)

        batch.max_input_length = max(batch.input_lengths)
        batch.max_decoder_input_length = max(batch.decoder_input_lengths)

        batch.decoder_attention_mask = batch.decoder_attention_mask[keep_indices] \
            if batch.decoder_attention_mask is not None else None
        batch.encoder_last_hidden_state = batch.encoder_last_hidden_state[keep_indices] \
            if batch.encoder_last_hidden_state is not None else None
        batch.input_ids = batch.input_ids[keep_indices]
        batch.attention_mask = batch.attention_mask[keep_indices]
        batch.decoder_input_ids = batch.decoder_input_ids[keep_indices]
        if batch.past_key_values is not None:
            batch.past_key_values = [
                [t[keep_indices] for t in layer] for layer in batch.past_key_values
            ]

        batch.size = new_size

        return batch


class Seq2SeqLM(Model):
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        tokenizer.bos_token_id = self.model.config.decoder_start_token_id

        super(Seq2SeqLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @property
    def batch_type(self) -> Type[Seq2SeqLMBatch]:
        return Seq2SeqLMBatch

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask: Optional,
        encoder_last_hidden_state: Optional,
        past_key_values: Optional = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_last_hidden_state,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return (
            outputs.logits,
            outputs.encoder_last_hidden_state,
            outputs.past_key_values,
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(self, batch: Seq2SeqLMBatch, prefill: bool = False) -> List[Generation]:
        if batch.decoder_attention_mask is not None:
            # slice to the correct shape
            decoder_attention_mask = batch.decoder_attention_mask[
                :, : -batch.padding_right_offset
            ]
        else:
            decoder_attention_mask = None

        # check if first forward or not
        if batch.past_key_values is not None:
            # Only take the last token
            decoder_input_ids = batch.decoder_input_ids[:, -1].unsqueeze(-1)
        else:
            decoder_input_ids = batch.decoder_input_ids

        # Wrap `encoder_last_hidden_state` because for some reason, Transformers does a `encoder_last_hidden_state[0]`
        # internally...
        if batch.encoder_last_hidden_state is not None:
            encoder_last_hidden_state = [batch.encoder_last_hidden_state]
        else:
            encoder_last_hidden_state = batch.encoder_last_hidden_state

        logits, encoder_last_hidden_state, past = self.forward(
            batch.input_ids,
            batch.attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_last_hidden_state,
            batch.past_key_values,
        )

        # New values for next forward
        next_batch_decoder_input_ids = []

        # Results
        generations: List[Generation] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            logits,
            batch.next_token_choosers,
            batch.decoder_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            logits,
            next_token_chooser,
            decoder_input_ids,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                decoder_input_ids.view(1, -1), logits
            )

            # Append next token to decoder tokens
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.squeeze(1)])

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()

            next_batch_decoder_input_ids.append(decoder_input_ids.unsqueeze(0))
            batch.decoder_input_lengths[i] += 1

            # Prefill
            if prefill:
                prefill_tokens = PrefillTokens(
                    [self.tokenizer.bos_token_id],
                    [float("nan")],
                )
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

        # Update decoder_attention_mask as we added a new token to decoder_input_ids
        if batch.decoder_attention_mask is not None:
            batch.decoder_attention_mask[:, -batch.padding_right_offset] = 1

        batch.input_ids = None
        batch.decoder_input_ids = torch.cat(next_batch_decoder_input_ids)
        batch.encoder_last_hidden_state = encoder_last_hidden_state
        batch.past_key_values = past
        batch.max_decoder_input_length += 1
        batch.padding_right_offset -= 1

        return generations
