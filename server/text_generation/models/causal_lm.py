import torch

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple, List, Type

from text_generation.models import Model
from text_generation.models.types import GeneratedText
from text_generation.pb import generate_pb2
from text_generation.utils import NextTokenChooser, StoppingCriteria


@dataclass
class CausalLMBatch:
    batch_id: int
    requests: List[generate_pb2.Request]

    # Decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # All tokens
    all_input_ids: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    size: int
    max_sequence_length: int

    def to_pb(self):
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=self.size,
        )

    @classmethod
    def from_pb(
        cls, pb: generate_pb2.Batch, tokenizer: AutoTokenizer, device: torch.device
    ) -> "CausalLMBatch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        input_lengths = []

        # Parse batch
        for r in pb.requests:
            inputs.append(r.inputs)
            input_lengths.append(r.input_length)
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

        tokenized_inputs = tokenizer(
            inputs, return_tensors="pt", padding=True, pad_to_multiple_of=8
        ).to(device)
        all_input_ids = tokenized_inputs["input_ids"].unsqueeze(-1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            past_key_values=None,
            all_input_ids=all_input_ids,
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
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

            # Slicing end index for this batch
            end_index = start_index + batch.size

            # We only concatenate batches that did at least one step
            if batch.input_ids.shape[1] > 1:
                raise ValueError("Batch input_ids should be of shape (batch_size, 1)")

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = torch.empty(
                    (total_batch_size, 1),
                    dtype=batch.input_ids.dtype,
                    device=batch.input_ids.device,
                )
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            # Create padded tensor
            if attention_mask is None:
                attention_mask = torch.zeros(
                    (total_batch_size, max_sequence_length),
                    dtype=batch.attention_mask.dtype,
                    device=batch.attention_mask.device,
                )

            # We need to slice the attention mask to remove padding from previous steps
            attention_mask[
                start_index:end_index, -batch.max_sequence_length :
            ] = batch.attention_mask[:, -batch.max_sequence_length :]

            for j, past in enumerate(batch.past_key_values):
                # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
                # BLOOM: [batch_size * num_heads, ...] vs [batch_size, num_heads, ...]
                head_dim, padded_sequence_length = past[0].shape[-2:]
                num_heads = (
                    past[0]
                    .view(batch.size, -1, head_dim, padded_sequence_length)
                    .shape[1]
                )

                # This will run only once per layer
                if j == len(past_key_values):
                    past_key_values.append([])

                # Decoder past
                for k, t in enumerate(past):
                    # Needed because BLOOM past shapes are not the same for keys and values
                    # Keys:   [batch_size * num_heads, head_dim, seq_length]
                    # Values: [batch_size * num_heads, seq_length, head_dim]
                    head_dim_last = False
                    if t.shape[-2] == head_dim:
                        t = t.view(
                            batch.size, num_heads, head_dim, padded_sequence_length
                        )
                        padded_t_shape = (
                            total_batch_size,
                            num_heads,
                            head_dim,
                            max_sequence_length - 1,
                        )
                    elif t.shape[-1] == head_dim:
                        head_dim_last = True
                        t = t.view(
                            batch.size, num_heads, padded_sequence_length, head_dim
                        )
                        padded_t_shape = (
                            total_batch_size,
                            num_heads,
                            max_sequence_length - 1,
                            head_dim,
                        )
                    else:
                        raise ValueError(f"shape {t.shape} is not valid")

                    # Initialize tensors
                    # This will run only once per layer and per past tensor
                    if k == len(past_key_values[j]):
                        past_key_values[j].append(
                            torch.zeros(padded_t_shape, dtype=t.dtype, device=t.device)
                        )

                    # We slice the past keys and values to remove the padding from previous batches
                    if not head_dim_last:
                        past_key_values[j][k][
                            start_index:end_index,
                            :,
                            :,
                            -(batch.max_sequence_length - 1) :,
                        ] = t[:, :, :, -(batch.max_sequence_length - 1) :]
                    else:
                        past_key_values[j][k][
                            start_index:end_index,
                            :,
                            -(batch.max_sequence_length - 1) :,
                            :,
                        ] = t[:, :, -(batch.max_sequence_length - 1) :, :]

            start_index += batch.size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=total_batch_size,
            max_sequence_length=max_sequence_length,
        )


class CausalLM(Model):
    def __init__(self, model_name: str, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize,
        ).eval()

        super(CausalLM, self).__init__(
            tokenizer=tokenizer,
            num_heads=self.model.config.num_attention_heads,
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
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_tokens,
        ) in enumerate(iterator):
            # Select next token
            next_token = next_token_chooser(all_tokens, logits.unsqueeze(0)[:, -1])

            # Append next token to all tokens
            all_tokens = torch.cat([all_tokens, next_token])

            # Evaluate stopping criteria
            if stopping_criteria(all_tokens):
                # Decode all tokens
                output = self.tokenizer.decode(
                    all_tokens.squeeze(-1), skip_special_tokens=True
                )
                # Add to the list of finished generations with the original request
                generated_texts.append(
                    GeneratedText(request, output, stopping_criteria.current_tokens)
                )
            # add to the next batch
            else:
                next_batch_keep_indices.append(i)
                next_batch_input_ids.append(next_token)
                next_batch_all_input_ids.append(all_tokens)
                next_batch_size += 1
                new_input_length = input_length + 1
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
                    t.view(-1, self.num_heads, *t.shape[-2:])[next_batch_keep_indices]
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
                torch.ones((next_batch_size, 1)).to(self.device),
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
            input_lengths=next_batch_input_lengths,
            next_token_choosers=next_batch_next_token_choosers,
            stopping_criterias=next_batch_stopping_criterias,
            size=next_batch_size,
            max_sequence_length=next_batch_max_sequence_length,
        )
        return generated_texts, next_batch
