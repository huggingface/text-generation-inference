import torch

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type

from text_generation.models import Model
from text_generation.models.types import GeneratedText, Batch
from text_generation.pb import generate_pb2
from text_generation.utils import NextTokenChooser, StoppingCriteria


@dataclass
class CausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # All tokens
    all_input_ids: List[torch.Tensor]
    all_logprobs: List[Optional[torch.Tensor]]

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    size: int
    max_sequence_length: int

    # Past metadata
    keys_head_dim_last: bool = True

    def to_pb(self) -> generate_pb2.Batch:
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=self.size,
        )

    @classmethod
    def from_pb(
        cls, pb: generate_pb2.Batch, tokenizer: PreTrainedTokenizerBase, device: torch.device
    ) -> "CausalLMBatch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        input_lengths = []
        all_logprobs = []

        # Parse batch
        for r in pb.requests:
            inputs.append(r.inputs)
            input_lengths.append(r.input_length)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters))
            stopping_criterias.append(
                StoppingCriteria.from_pb(r.stopping_parameters, tokenizer)
            )
            all_logprobs.append(None)

        pad_to_multiple_of = 8 if device.type == "cuda" else None
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
        ).to(device)
        all_input_ids = tokenized_inputs["input_ids"].unsqueeze(-1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            past_key_values=None,
            all_input_ids=all_input_ids,
            all_logprobs=all_logprobs,
            input_lengths=input_lengths,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=pb.size,
            max_sequence_length=max(input_lengths),
        )

    @classmethod
    def concatenate(cls, batches: List["CausalLMBatch"]) -> "CausalLMBatch":
        # Used for padding
        total_batch_size = sum(batch.size for batch in batches)
        max_sequence_length = max(batch.max_sequence_length for batch in batches)

        # Batch attributes
        requests = []
        input_lengths = []
        all_input_ids = []
        all_logprobs = []
        next_token_choosers = []
        stopping_criterias = []

        # Batch tensors
        input_ids = None
        attention_mask = None
        past_key_values = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            all_input_ids.extend(batch.all_input_ids)
            all_logprobs.extend(batch.all_logprobs)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

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
                    (total_batch_size, max_sequence_length),
                )

            # We need to slice the attention mask to remove padding from previous steps
            attention_mask[
                start_index:end_index, -batch.max_sequence_length :
            ] = batch.attention_mask[:, -batch.max_sequence_length :]

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
                    max_sequence_length - 1,
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
                        max_sequence_length - 1,
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
                        -(batch.max_sequence_length - 1) :,
                        :,
                    ] = past_keys[:, :, -(batch.max_sequence_length - 1) :, :]
                else:
                    past_key_values[j][0][
                        start_index:end_index,
                        :,
                        :,
                        -(batch.max_sequence_length - 1) :,
                    ] = past_keys[:, :, :, -(batch.max_sequence_length - 1) :]

                past_key_values[j][1][
                    start_index:end_index, :, -(batch.max_sequence_length - 1) :, :
                ] = past_values[:, :, -(batch.max_sequence_length - 1) :, :]

            start_index += batch.size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            all_input_ids=all_input_ids,
            all_logprobs=all_logprobs,
            input_lengths=input_lengths,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=total_batch_size,
            max_sequence_length=max_sequence_length,
            keys_head_dim_last=batches[0].keys_head_dim_last,
        )


class CausalLM(Model):
    def __init__(self, model_name: str, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return CausalLMBatch

    def forward(
        self, input_ids, attention_mask, past_key_values: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    def generate_token(
        self, batch: CausalLMBatch
    ) -> Tuple[List[GeneratedText], Optional[CausalLMBatch]]:
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        context_manager = (
            torch.no_grad if self.device.type == "cpu" else torch.inference_mode
        )
        with context_manager():
            logits, past = self.forward(
                batch.input_ids, batch.attention_mask, batch.past_key_values
            )

        # List of indices to cache
        next_batch_keep_indices = []

        # New values for next forward
        next_batch_input_lengths = []
        next_batch_input_ids = []
        next_batch_all_input_ids = []
        next_batch_all_logprobs = []

        # Metadata
        next_batch_size = 0
        next_batch_max_sequence_length = 0

        # Finished requests
        generated_texts: List[GeneratedText] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.all_logprobs,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
            all_logprobs,
        ) in enumerate(iterator):
            # Select next token
            tokens, logprobs = next_token_chooser(all_input_ids, logits)
            next_token = tokens[-1].view(1, 1)

            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token])
            new_input_length = input_length + 1

            if all_logprobs is None:
                # logprobs of all prompt tokens (except the first one) and the generated token
                all_logprobs = logprobs.gather(1, all_input_ids[1:])
            else:
                # logprob of the generated token
                next_token_logprob = logprobs[-1, next_token]
                all_logprobs = torch.cat([all_logprobs, next_token_logprob])

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token.squeeze(),
                self.tokenizer.decode(
                    next_token.squeeze(), clean_up_tokenization_spaces=False
                ),
            )
            if stop:
                # Decode all tokens
                output_text = self.tokenizer.decode(
                    all_input_ids.squeeze(-1), skip_special_tokens=True,
                    cleanup_tokenization_spaces=False
                )
                # Slice with input_length to remove padding
                token_ids = all_input_ids[-new_input_length:]
                tokens = self.tokenizer.batch_decode(token_ids)
                # Add NaN for the first prompt token
                logprobs = [float("nan")] + all_logprobs[-new_input_length:].squeeze(
                    1
                ).tolist()

                # Add to the list of finished generations with the original request
                generated_texts.append(
                    GeneratedText(
                        request=request,
                        output_text=output_text,
                        generated_tokens=stopping_criteria.current_tokens,
                        tokens=tokens,
                        token_ids=token_ids.squeeze(1).tolist(),
                        logprobs=logprobs,
                        reason=reason,
                    )
                )
            # add to the next batch
            else:
                next_batch_keep_indices.append(i)
                next_batch_input_ids.append(next_token)
                next_batch_all_input_ids.append(all_input_ids)
                next_batch_all_logprobs.append(all_logprobs)
                next_batch_size += 1
                next_batch_input_lengths.append(new_input_length)
                next_batch_max_sequence_length = max(
                    next_batch_max_sequence_length, new_input_length
                )

        # We finished all generations in the batch; there is no next batch
        if not next_batch_keep_indices:
            return generated_texts, None

        next_batch_input_ids = torch.cat(next_batch_input_ids, dim=0)
        # If we finished at least one generation, we need to evict the indices of the generations that finished
        # from the values of the next batch
        if generated_texts:
            # Apply indices to attention mask, past key values and other items that need to be cached
            next_batch_attention_mask = batch.attention_mask[next_batch_keep_indices]
            # Force past to be of dim [batch_size, num_heads, ...] for easy indexing
            next_batch_past_key_values = [
                [
                    t.view(batch.size, -1, *t.shape[-2:])[next_batch_keep_indices]
                    for t in layer
                ]
                for layer in past
            ]
            next_batch_requests = [batch.requests[i] for i in next_batch_keep_indices]
            next_batch_next_token_choosers = [
                batch.next_token_choosers[i] for i in next_batch_keep_indices
            ]
            next_batch_stopping_criterias = [
                batch.stopping_criterias[i] for i in next_batch_keep_indices
            ]
        else:
            next_batch_attention_mask = batch.attention_mask
            next_batch_past_key_values = past
            next_batch_requests = batch.requests
            next_batch_next_token_choosers = batch.next_token_choosers
            next_batch_stopping_criterias = batch.stopping_criterias

        # Update attention_mask with padding as we added a new token to input_ids
        next_batch_attention_mask = torch.cat(
            [
                next_batch_attention_mask,
                next_batch_attention_mask.new_ones(next_batch_size, 1),
            ],
            dim=1,
        )

        next_batch = CausalLMBatch(
            batch_id=batch.batch_id,
            requests=next_batch_requests,
            input_ids=next_batch_input_ids,
            attention_mask=next_batch_attention_mask,
            past_key_values=next_batch_past_key_values,
            all_input_ids=next_batch_all_input_ids,
            all_logprobs=next_batch_all_logprobs,
            input_lengths=next_batch_input_lengths,
            next_token_choosers=next_batch_next_token_choosers,
            stopping_criterias=next_batch_stopping_criterias,
            size=next_batch_size,
            max_sequence_length=next_batch_max_sequence_length,
            keys_head_dim_last=batch.keys_head_dim_last,
        )
        return generated_texts, next_batch
