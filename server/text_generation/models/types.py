import torch

from abc import abstractmethod
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
    @abstractmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        raise NotImplementedError


@dataclass
class GeneratedText:
    request: generate_pb2.Request
    output: str

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(request=self.request, output=self.output)
