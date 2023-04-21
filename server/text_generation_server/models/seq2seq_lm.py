import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict

from text_generation_server.models import Model
from text_generation_server.models.types import (
    GeneratedText,
    Batch,
    Generation,
    PrefillTokens,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling

tracer = trace.get_tracer(__name__)


@dataclass
class Seq2SeqLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Encoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    # Decoder values
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: Optional[torch.Tensor]
    encoder_last_hidden_state: Optional[torch.Tensor]

    # All tokens
    all_decoder_input_ids: List[torch.Tensor]

    # Seq2SeqLM keeps track of both encoder and decoder attention keys and values
    past_key_values: Optional[List[Tuple]]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    decoder_input_lengths: List[int]
    offsets: List[Optional[int]]
    token_offsets: List[Optional[int]]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    max_input_length: int
    max_decoder_input_length: int
    padding_right_offset: int

    def to_pb(self) -> generate_pb2.Batch:
        """Convert a Seq2SeqLMBatch to a text_generation_server.v1.Batch protobuf"""
        return generate_pb2.Batch(
            id=self.batch_id, requests=self.requests, size=len(self)
        )

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
        stopping_criterias = []

        decoder_input_lengths = []
        offsets = []
        token_offsets = []
        requests_idx_mapping = {}

        # Parse batch
        max_truncation = 0
        padding_right_offset = 0
        for i, r in enumerate(pb.requests):
            inputs.append(r.inputs)
            requests_idx_mapping[r.id] = i
            decoder_input_lengths.append(1)
            offsets.append(None)
            token_offsets.append(None)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            max_truncation = max(max_truncation, r.truncate)
            padding_right_offset = max(
                padding_right_offset, stopping_criteria.max_new_tokens
            )

        # Tokenize batch
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max_truncation,
        ).to(device)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()

        # Decoder sequence only contains the bos_token
        decoder_input_ids = (
            torch.tensor(tokenizer.bos_token_id, device=device)
            .repeat(len(pb.requests))
            .view(-1, 1)
        )
        all_decoder_input_ids = decoder_input_ids.view(-1).split(1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            all_decoder_input_ids=list(all_decoder_input_ids),
            decoder_attention_mask=None,
            encoder_last_hidden_state=None,
            past_key_values=None,
            input_lengths=input_lengths.tolist(),
            decoder_input_lengths=decoder_input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length.item(),
            max_decoder_input_length=1,
            padding_right_offset=padding_right_offset,
        )

    @tracer.start_as_current_span("filter")
    def filter(
        self, requests: List[generate_pb2.Request]
    ) -> Optional["Seq2SeqLMBatch"]:
        if len(requests) == 0:
            raise ValueError("Batch must have at least one request")
        if len(requests) == len(self):
            return self

        keep_indices = []

        # New values after filtering
        requests_idx_mapping = {}
        input_lengths = []
        decoder_input_lengths = []
        offsets = []
        token_offsets = []

        all_decoder_input_ids = []

        next_token_choosers = []
        stopping_criterias = []

        max_input_length = 0
        max_decoder_input_length = 0

        for i, r in enumerate(requests):
            idx = self.requests_idx_mapping[r.id]
            requests_idx_mapping[r.id] = i
            keep_indices.append(idx)

            offsets.append(self.offsets[idx])
            token_offsets.append(self.token_offsets[idx])

            all_decoder_input_ids.append(self.all_decoder_input_ids[idx])

            request_input_length = self.input_lengths[idx]
            input_lengths.append(request_input_length)
            max_input_length = max(max_input_length, request_input_length)

            request_decoder_input_length = self.decoder_input_lengths[idx]
            decoder_input_lengths.append(request_decoder_input_length)
            max_decoder_input_length = max(
                max_decoder_input_length, request_decoder_input_length
            )

            next_token_choosers.append(self.next_token_choosers[idx])
            stopping_criterias.append(self.stopping_criterias[idx])

        # Apply indices to input_ids, attention mask, past key values and other items that need to be cached
        decoder_input_ids = self.decoder_input_ids[keep_indices]
        attention_mask = self.attention_mask[keep_indices]
        if self.decoder_attention_mask is not None:
            decoder_attention_mask = self.decoder_attention_mask[keep_indices]
        else:
            decoder_attention_mask = None

        encoder_last_hidden_state = self.encoder_last_hidden_state[keep_indices]

        past_key_values = [
            [t[keep_indices] for t in layer] for layer in self.past_key_values
        ]

        return Seq2SeqLMBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            all_decoder_input_ids=all_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_last_hidden_state=encoder_last_hidden_state,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length,
            max_decoder_input_length=max_decoder_input_length,
            padding_right_offset=self.padding_right_offset,
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
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            max_decoder_input_length = max(
                max_decoder_input_length, batch.max_decoder_input_length
            )
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        requests_idx_mapping = {}
        all_decoder_input_ids = []
        input_lengths = []
        decoder_input_lengths = []
        offsets = []
        token_offsets = []
        next_token_choosers = []
        stopping_criterias = []

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
            all_decoder_input_ids.extend(batch.all_decoder_input_ids)
            input_lengths.extend(batch.input_lengths)
            decoder_input_lengths.extend(batch.decoder_input_lengths)
            offsets.extend(batch.offsets)
            token_offsets.extend(batch.token_offsets)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index

            # Slicing end index for this batch
            end_index = start_index + len(batch)

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
                    (total_batch_size, 1),
                )
            # Copy to correct indices
            decoder_input_ids[start_index:end_index] = batch.decoder_input_ids

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

            start_index += len(batch)

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            all_decoder_input_ids=all_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_last_hidden_state=encoder_last_hidden_state,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length,
            max_decoder_input_length=max_decoder_input_length,
            padding_right_offset=padding_right_offset,
        )

    def __len__(self):
        return len(self.requests)


class Seq2SeqLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: bool = False,
        decode_buffer: int = 3,
    ):
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
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )
        tokenizer.bos_token_id = self.model.config.decoder_start_token_id

        super(Seq2SeqLM, self).__init__(
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            decode_buffer=decode_buffer,
        )

    @property
    def batch_type(self) -> Type[Seq2SeqLMBatch]:
        return Seq2SeqLMBatch

    def decode(self, decoder_ids: List[int]) -> str:
        return self.tokenizer.decode(
            decoder_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

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
    def generate_token(
        self, batch: Seq2SeqLMBatch
    ) -> Tuple[List[Generation], Optional[Seq2SeqLMBatch]]:
        if batch.decoder_attention_mask is not None:
            # slice to the correct shape
            decoder_attention_mask = batch.decoder_attention_mask[
                :, : -batch.padding_right_offset
            ]
        else:
            decoder_attention_mask = None

        # Wrap `encoder_last_hidden_state` because for some reason, Transformers does a `encoder_last_hidden_state[0]`
        # internally...
        if batch.encoder_last_hidden_state is not None:
            encoder_last_hidden_state = [batch.encoder_last_hidden_state]
        else:
            encoder_last_hidden_state = None

        logits, encoder_last_hidden_state, past = self.forward(
            batch.input_ids,
            batch.attention_mask,
            batch.decoder_input_ids,
            decoder_attention_mask,
            encoder_last_hidden_state,
            batch.past_key_values,
        )

        # Finished requests
        generations: List[Generation] = []
        stopped = True

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.offsets,
            batch.token_offsets,
            batch.decoder_input_lengths,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_decoder_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            offset,
            token_offset,
            decoder_input_length,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_decoder_input_ids,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_decoder_input_ids.view(1, -1), logits
            )

            # Append next token to decoder tokens
            all_decoder_input_ids = torch.cat(
                [all_decoder_input_ids, next_token_id.squeeze(1)]
            )
            new_decoder_input_length = decoder_input_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, offset, token_offset = self.decode_token(
                all_decoder_input_ids, offset, token_offset
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(next_token_id, next_token_text)

            if stop:
                # Slice with decoder_input_length to remove padding
                # Decode all tokens
                output_text = self.decode(all_decoder_input_ids[-decoder_input_length:])

                # Get seed
                if isinstance(next_token_chooser.choice, Sampling):
                    seed = next_token_chooser.choice.seed
                else:
                    seed = None

                generated_text = GeneratedText(
                    output_text, stopping_criteria.current_tokens, reason, seed
                )
            else:
                # Keep request in the batch
                generated_text = None
                stopped = False

            # Prefill
            if stopping_criteria.current_tokens == 1:
                prefill_tokens = PrefillTokens(
                    [self.tokenizer.bos_token_id],
                    [float("nan")],
                    [self.tokenizer.bos_token],
                )
            else:
                prefill_tokens = None

            generation = Generation(
                request.id,
                prefill_tokens,
                next_token_id_squeezed,
                next_token_logprob,
                next_token_text,
                next_token_id_squeezed.item() in self.all_special_ids,
                generated_text,
            )

            generations.append(generation)

            # Update values
            batch.decoder_input_ids[i] = next_token_id
            batch.all_decoder_input_ids[i] = all_decoder_input_ids
            batch.input_lengths[i] = input_length
            batch.decoder_input_lengths[i] = new_decoder_input_length
            batch.offsets[i] = offset
            batch.token_offsets[i] = token_offset
            batch.max_input_length = max(batch.max_input_length, input_length)
            batch.max_decoder_input_length = max(
                batch.max_decoder_input_length, new_decoder_input_length
            )

        # We finished all generations in the batch; there is no next batch
        if stopped:
            return generations, None

        # We don't need input_ids after the prefill forward
        batch.input_ids = None
        batch.encoder_last_hidden_state = encoder_last_hidden_state
        batch.past_key_values = past
        # Update decoder_attention_mask as we added a new token to input_ids
        if batch.decoder_attention_mask is not None:
            batch.decoder_attention_mask[:, -batch.padding_right_offset] = 1
        batch.padding_right_offset -= 1

        return generations, batch
