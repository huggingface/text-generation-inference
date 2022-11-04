import torch

from dataclasses import dataclass
from typing import List, Dict

from transformers import AutoTokenizer

from text_generation.pb import generate_pb2
from text_generation.utils import NextTokenChooser, StoppingCriteria


@dataclass
class Batch:
    batch_id: int
    requests: List[generate_pb2.Request]
    all_input_lengths: List[int]
    input_ids: Dict[str, torch.Tensor]
    all_input_ids: List[torch.Tensor]
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]
    size: int
    max_sequence_length: int

    def to_pb(self):
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=self.size,
            max_sequence_length=self.max_sequence_length,
        )

    @classmethod
    def from_pb(
            cls, pb: generate_pb2.Batch, tokenizer: AutoTokenizer, device: torch.device
    ) -> "Batch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        all_input_lengths = []

        # Parse batch
        for r in pb.requests:
            inputs.append(r.inputs)
            all_input_lengths.append(r.input_length)
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

        input_ids = tokenizer(
            inputs, return_tensors="pt", padding=True, pad_to_multiple_of=8
        ).to(device)
        all_input_ids = input_ids["input_ids"].unsqueeze(-1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            all_input_lengths=all_input_lengths,
            input_ids=input_ids,
            all_input_ids=all_input_ids,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=pb.size,
            max_sequence_length=pb.max_sequence_length,
        )

    @classmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        # Used for padding
        total_batch_size = sum(batch.size for batch in batches)
        max_sequence_length = max(batch.max_sequence_length for batch in batches)
        # Only needed for Seq2SeqLM
        max_encoded_sequence_length = None

        # Batch attributes
        input_ids = {"input_ids": None, "attention_mask": None, "past_key_values": []}
        requests = []
        all_input_lengths = []
        all_input_ids = []
        next_token_choosers = []
        stopping_criterias = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            all_input_lengths.extend(batch.all_input_lengths)
            all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

            # Slicing end index for this batch
            end_index = start_index + batch.size

            # We only concatenate batches that did at least one step
            if batch.input_ids["input_ids"].shape[1] > 1:
                raise ValueError("Batch input_ids should be of shape (batch_size, 1)")

            # Initialize tensors
            if i == 0:
                input_ids["input_ids"] = torch.empty(
                    (total_batch_size, 1),
                    dtype=batch.input_ids["input_ids"].dtype,
                    device=batch.input_ids["input_ids"].device,
                )
                input_ids["attention_mask"] = torch.zeros(
                    (total_batch_size, max_sequence_length),
                    dtype=batch.input_ids["attention_mask"].dtype,
                    device=batch.input_ids["attention_mask"].device,
                )

            # input_ids["input_ids"] is always of shape [batch_size, 1]
            # We do not need to pad it
            input_ids["input_ids"][start_index:end_index] = batch.input_ids["input_ids"]

            # We need to slice the attention mask to remove padding from previous steps
            input_ids["attention_mask"][
            start_index:end_index, -batch.max_sequence_length:
            ] = batch.input_ids["attention_mask"][:, -batch.max_sequence_length:]

            for j, past in enumerate(batch.input_ids["past_key_values"]):
                # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
                # BLOOM: [batch_size * num_heads, ...] vs [batch_size, num_heads, ...]
                head_dim, padded_sequence_length = past[0].shape[-2:]
                num_heads = (
                    past[0]
                    .view(batch.size, -1, head_dim, padded_sequence_length)
                    .shape[1]
                )

                # This will run only once per layer
                if j == len(input_ids["past_key_values"]):
                    input_ids["past_key_values"].append([])

                # Decoder past
                for k, t in enumerate(past[:2]):
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
                    if k == len(input_ids["past_key_values"][j]):
                        input_ids["past_key_values"][j].append(
                            torch.zeros(padded_t_shape, dtype=t.dtype, device=t.device)
                        )

                    # We slice the past keys and values to remove the padding from previous batches
                    if not head_dim_last:
                        input_ids["past_key_values"][j][k][
                        start_index:end_index,
                        :,
                        :,
                        -(batch.max_sequence_length - 1):,
                        ] = t[:, :, :, -(batch.max_sequence_length - 1):]
                    else:
                        input_ids["past_key_values"][j][k][
                        start_index:end_index,
                        :,
                        -(batch.max_sequence_length - 1):,
                        :,
                        ] = t[:, :, -(batch.max_sequence_length - 1):, :]

                # Seq2SeqLM specific past (encoder past)
                for k, t in enumerate(past[2:]):
                    if max_encoded_sequence_length is None:
                        max_encoded_sequence_length = max(max(batch.all_input_lengths) for batch in batches)
                    batch_max_encoded_sequence_length = max(batch.all_input_lengths)

                    padded_t_shape = (total_batch_size, num_heads, max_encoded_sequence_length, head_dim)

                    idx = k + 2

                    # Initialize tensors
                    # This will run only once per layer and per past tensor
                    if idx == len(input_ids["past_key_values"][j]):
                        input_ids["past_key_values"][j].append(
                            torch.zeros(padded_t_shape, dtype=t.dtype, device=t.device)
                        )

                    input_ids["past_key_values"][j][idx][
                    start_index:end_index,
                    :,
                    -batch_max_encoded_sequence_length:,
                    :
                    ] = t[:, :, -batch_max_encoded_sequence_length:, :]

            start_index += batch.size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            all_input_lengths=all_input_lengths,
            input_ids=input_ids,
            all_input_ids=all_input_ids,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=total_batch_size,
            max_sequence_length=max_sequence_length,
        )


@dataclass
class GeneratedText:
    request: generate_pb2.Request
    output: str
    tokens: int

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(request=self.request, output=self.output, tokens=self.tokens)
