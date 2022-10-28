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
            stopping_criterias.append(StoppingCriteria(max_new_tokens=r.max_new_tokens))

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
                start_index:end_index, -batch.max_sequence_length :
            ] = batch.input_ids["attention_mask"][:, -batch.max_sequence_length :]

            for j, past in enumerate(batch.input_ids["past_key_values"]):
                past_keys = past[0]
                past_values = past[1]

                _, head_dim, padded_sequence_length = past_keys.shape

                # Reshape the tensors to make slicing easier
                past_keys = past_keys.view(
                    batch.size, -1, head_dim, padded_sequence_length
                )
                past_values = past_values.view(
                    batch.size, -1, padded_sequence_length, head_dim
                )
                num_heads = past_keys.shape[1]

                # Initialize tensors
                # This will run only once per layer
                if j == len(input_ids["past_key_values"]):
                    padded_past_keys = torch.zeros(
                        (
                            total_batch_size,
                            num_heads,
                            head_dim,
                            max_sequence_length - 1,
                        ),
                        dtype=past_keys.dtype,
                        device=past_keys.device,
                    )
                    padded_past_values = torch.zeros(
                        (
                            total_batch_size,
                            num_heads,
                            max_sequence_length - 1,
                            head_dim,
                        ),
                        dtype=past_values.dtype,
                        device=past_values.device,
                    )
                    input_ids["past_key_values"].append(
                        [padded_past_keys, padded_past_values]
                    )

                # We slice the past keys and values to remove the padding from previous batches
                input_ids["past_key_values"][j][0][
                    start_index:end_index, :, :, -(batch.max_sequence_length - 1) :
                ] = past_keys[:, :, :, -(batch.max_sequence_length - 1) :]

                input_ids["past_key_values"][j][1][
                    start_index:end_index, :, -(batch.max_sequence_length - 1) :, :
                ] = past_values[:, :, -(batch.max_sequence_length - 1) :, :]

                # If we are on the last batch, we need to reshape the tensors
                if (i + 1) == len(batches):
                    input_ids["past_key_values"][j][0] = input_ids["past_key_values"][
                        j
                    ][0].view(total_batch_size * num_heads, head_dim, -1)
                    input_ids["past_key_values"][j][1] = input_ids["past_key_values"][
                        j
                    ][1].view(total_batch_size * num_heads, -1, head_dim)

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

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(request=self.request, output=self.output)
