import torch

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Tuple, List, Type

from text_generation.models import Model
from text_generation.models.types import GeneratedText
from text_generation.pb import generate_pb2
from text_generation.utils import NextTokenChooser, StoppingCriteria


@dataclass
class Seq2SeqLMBatch:
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
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    size: int
    max_input_length: int
    max_decoder_input_length: int

    def to_pb(self):
        """Convert a Seq2SeqLMBatch to a text_generation.v1.Batch protobuf"""
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=self.size,
        )

    @classmethod
    def from_pb(
        cls, pb: generate_pb2.Batch, tokenizer: AutoTokenizer, device: torch.device
    ) -> "Seq2SeqLMBatch":
        """Convert a text_generation.v1.Batch protobuf to a Seq2SeqLMBatch"""
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        input_lengths = []

        decoder_input_ids = []
        decoder_input_lengths = []

        # Parse batch
        for r in pb.requests:
            inputs.append(r.inputs)
            input_lengths.append(r.input_length)
            # Decoder sequence only contains the bos_token
            decoder_input_ids.append(tokenizer.bos_token_id)
            decoder_input_lengths.append(1)
            next_token_choosers.append(
                NextTokenChooser(
                    temperature=r.parameters.temperature,
                    top_k=r.parameters.top_k,
                    top_p=r.parameters.top_p,
                    do_sample=r.parameters.do_sample,
                )
            )
            stopping_criterias.append(
                StoppingCriteria(
                    eos_token_id=tokenizer.eos_token_id, max_new_tokens=r.max_new_tokens
                )
            )

        # Tokenize batch
        pad_to_multiple_of = 8 if "gpu" in str(device) else None
        tokenized_inputs = tokenizer(
            inputs, return_tensors="pt", padding=True, pad_to_multiple_of=pad_to_multiple_of
        ).to(device)
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
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=len(pb.requests),
            max_input_length=max(input_lengths),
            max_decoder_input_length=1,
        )

    @classmethod
    def concatenate(cls, batches: List["Seq2SeqLMBatch"]) -> "Seq2SeqLMBatch":
        """Concatenate multiple batches together by padding internal torch tensors"""

        # Used for padding
        total_batch_size = sum(batch.size for batch in batches)
        max_input_length = max(batch.max_input_length for batch in batches)
        max_decoder_input_length = max(
            batch.max_decoder_input_length for batch in batches
        )

        # Batch attributes
        requests = []
        input_lengths = []
        decoder_input_lengths = []
        next_token_choosers = []
        stopping_criterias = []

        # Batch tensors
        input_ids = None
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
            stopping_criterias.extend(batch.stopping_criterias)

            # Slicing end index for this batch
            end_index = start_index + batch.size

            # We only concatenate batches that did at least one step
            if batch.encoder_last_hidden_state is None:
                raise ValueError("Batch encoder_last_hidden_state cannot be None")

            # Create padded tensor
            if input_ids is None:
                input_ids = torch.zeros(
                    (total_batch_size, max_input_length),
                    dtype=batch.input_ids.dtype,
                    device=batch.input_ids.device,
                )
            # Copy to correct indices
            input_ids[
                start_index:end_index, -batch.max_input_length :
            ] = batch.input_ids[:, -batch.max_input_length :]

            # Create padded tensor
            if attention_mask is None:
                attention_mask = torch.zeros(
                    (total_batch_size, max_input_length),
                    dtype=batch.attention_mask.dtype,
                    device=batch.attention_mask.device,
                )
            # Copy to correct indices
            attention_mask[
                start_index:end_index, -batch.max_input_length :
            ] = batch.attention_mask[:, -batch.max_input_length :]

            # Create padded tensor
            if decoder_input_ids is None:
                decoder_input_ids = torch.zeros(
                    (total_batch_size, max_decoder_input_length),
                    dtype=batch.decoder_input_ids.dtype,
                    device=batch.decoder_input_ids.device,
                )
            # Copy to correct indices
            decoder_input_ids[
                start_index:end_index, -batch.max_decoder_input_length :
            ] = batch.decoder_input_ids[:, -batch.max_decoder_input_length :]

            # Create padded tensor
            if decoder_attention_mask is None:
                decoder_attention_mask = torch.zeros(
                    (total_batch_size, max_decoder_input_length),
                    dtype=batch.attention_mask.dtype,  # As decoder_attention_mask might not exist,
                    device=batch.attention_mask.device,  # we use `batch.attention_maks` for device here
                )
            # If the decoder mask does not exist yet, all generations started at the same time and we never concatenated
            # this batch. All generations are of length `batch.max_decoder_input_length`.
            if batch.decoder_attention_mask is None:
                decoder_attention_mask[
                    start_index:end_index, -batch.max_decoder_input_length :
                ] = 1
            # If it exists, we need to index
            else:
                decoder_attention_mask[
                    start_index:end_index, -batch.max_decoder_input_length :
                ] = batch.decoder_attention_mask[:, -batch.max_decoder_input_length :]

            # Create padded tensor
            if encoder_last_hidden_state is None:
                encoder_last_hidden_state = torch.zeros(
                    (
                        total_batch_size,
                        max_input_length,
                        batch.encoder_last_hidden_state.shape[-1],
                    ),
                    dtype=batch.encoder_last_hidden_state.dtype,
                    device=batch.encoder_last_hidden_state.device,
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
                        past_key_values[j].append(
                            torch.zeros(padded_t_shape, dtype=t.dtype, device=t.device)
                        )

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
                        past_key_values[j].append(
                            torch.zeros(padded_t_shape, dtype=t.dtype, device=t.device)
                        )

                    past_key_values[j][idx][
                        start_index:end_index, :, -batch.max_input_length :, :
                    ] = t[:, :, -batch.max_input_length :, :]

            start_index += batch.size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_last_hidden_state=encoder_last_hidden_state,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=total_batch_size,
            max_input_length=max_input_length,
            max_decoder_input_length=max_decoder_input_length,
        )


class Seq2SeqLM(Model):
    def __init__(self, model_name: str, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
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
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1].unsqueeze(-1)

        # Wrap `encoder_last_hidden_state` because for some reason, Transformers does a `encoder_last_hidden_state[0]`
        # internally...
        if encoder_last_hidden_state is not None:
            encoder_last_hidden_state = [encoder_last_hidden_state]

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

    def generate_token(
        self, batch: Seq2SeqLMBatch
    ) -> Tuple[List[GeneratedText], Optional[Seq2SeqLMBatch]]:
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        context_manager = (
            torch.no_grad if self.device.type == "cpu" else torch.inference_mode
        )
        with context_manager():
            logits, encoder_last_hidden_state, past = self.forward(
                batch.input_ids,
                batch.attention_mask,
                batch.decoder_input_ids,
                batch.decoder_attention_mask,
                batch.encoder_last_hidden_state,
                batch.past_key_values,
            )

        # List of indices to cache
        next_batch_keep_indices = []

        # New values for next forward
        next_batch_input_lengths = []
        next_batch_decoder_input_ids = []
        next_batch_decoder_input_lengths = []

        # Metadata
        next_batch_size = 0
        next_batch_max_input_length = 0
        next_batch_max_decoder_input_length = 0

        # Finished requests
        generated_texts: List[GeneratedText] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.decoder_input_lengths,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.input_ids,
            batch.decoder_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            decoder_input_length,
            logits,
            next_token_chooser,
            stopping_criteria,
            input_tokens,
            decoder_tokens,
        ) in enumerate(iterator):
            all_tokens = torch.cat([input_tokens, decoder_tokens])
            # Select next token
            next_token = next_token_chooser(all_tokens, logits.unsqueeze(0)[:, -1])

            # Append next token to decoder tokens
            decoder_tokens = torch.cat([decoder_tokens, next_token.squeeze(1)])

            # Evaluate stopping criteria
            if stopping_criteria(decoder_tokens):
                # Decode tokens
                output = self.tokenizer.decode(decoder_tokens, skip_special_tokens=True)
                # Add to the list of finished generations with the original request
                generated_texts.append(
                    GeneratedText(request, output, stopping_criteria.current_tokens)
                )
            # add to the next batch
            else:
                next_batch_keep_indices.append(i)
                next_batch_decoder_input_ids.append(decoder_tokens.unsqueeze(0))
                next_batch_size += 1
                new_decoder_input_length = decoder_input_length + 1
                next_batch_input_lengths.append(input_length)
                next_batch_decoder_input_lengths.append(new_decoder_input_length)
                next_batch_max_input_length = max(
                    next_batch_max_input_length, input_length
                )
                next_batch_max_decoder_input_length = max(
                    next_batch_max_decoder_input_length, new_decoder_input_length
                )

        # We finished all generations in the batch; there is no next batch
        if not next_batch_keep_indices:
            return generated_texts, None

        next_batch_decoder_input_ids = torch.cat(next_batch_decoder_input_ids)
        # If we finished at least one generation, we need to evict the indices of the generations that finished
        # from the values of the next batch
        if generated_texts:
            # Apply indices to attention mask, past key values and other items that need to be cached
            next_batch_input_ids = batch.input_ids[next_batch_keep_indices]
            next_batch_attention_mask = batch.attention_mask[next_batch_keep_indices]

            if batch.decoder_attention_mask is not None:
                next_batch_decoder_attention_mask = batch.decoder_attention_mask[
                    next_batch_keep_indices
                ]
            else:
                next_batch_decoder_attention_mask = None

            next_batch_encoder_last_hidden_state = encoder_last_hidden_state[
                next_batch_keep_indices
            ]

            next_batch_past_key_values = [
                [t[next_batch_keep_indices] for t in layer] for layer in past
            ]
            next_batch_requests = [batch.requests[i] for i in next_batch_keep_indices]
            next_batch_next_token_choosers = [
                batch.next_token_choosers[i] for i in next_batch_keep_indices
            ]
            next_batch_stopping_criterias = [
                batch.stopping_criterias[i] for i in next_batch_keep_indices
            ]
        else:
            next_batch_input_ids = batch.input_ids
            next_batch_attention_mask = batch.attention_mask
            next_batch_decoder_attention_mask = batch.decoder_attention_mask
            next_batch_encoder_last_hidden_state = encoder_last_hidden_state
            next_batch_past_key_values = past

            next_batch_requests = batch.requests
            next_batch_next_token_choosers = batch.next_token_choosers
            next_batch_stopping_criterias = batch.stopping_criterias

        # Update decoder_attention_mask with padding as we added a new token to input_ids
        if next_batch_decoder_attention_mask is not None:
            next_batch_decoder_attention_mask = torch.cat(
                [
                    next_batch_decoder_attention_mask,
                    next_batch_decoder_attention_mask.new_ones(next_batch_size, 1),
                ],
                dim=1,
            )

        next_batch = Seq2SeqLMBatch(
            batch_id=batch.batch_id,
            requests=next_batch_requests,
            input_ids=next_batch_input_ids,
            attention_mask=next_batch_attention_mask,
            decoder_input_ids=next_batch_decoder_input_ids,
            decoder_attention_mask=next_batch_decoder_attention_mask,
            encoder_last_hidden_state=next_batch_encoder_last_hidden_state,
            past_key_values=next_batch_past_key_values,
            input_lengths=next_batch_input_lengths,
            decoder_input_lengths=next_batch_decoder_input_lengths,
            next_token_choosers=next_batch_next_token_choosers,
            stopping_criterias=next_batch_stopping_criterias,
            size=next_batch_size,
            max_input_length=next_batch_max_input_length,
            max_decoder_input_length=next_batch_max_decoder_input_length,
        )
        return generated_texts, next_batch
