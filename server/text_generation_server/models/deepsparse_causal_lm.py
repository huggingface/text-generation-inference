import numpy, torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Type, Dict

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
)

from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling

DEEPSPARSE_SEQUENCE_LENGTH = 128

@dataclass
class DeepSparseCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]
    
    # TODO: update to handle calculating max_tokens --- needed for CachedBatch

    # Decoder values
    input_ids_list: List[numpy.ndarray]
    past_key_values_list: Optional[List[DeepSparsePastKeyValues]]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
        )
    
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "DeepSparseCausalLMBatch":
        
        # parse batch
        input_ids_list = []
        next_token_choosers = []
        stopping_criterias = []
        requests_idx_mapping = {}

        # setup tokenizer for deepsparse left padding
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        padding, truncation = "longest", False

        # loop through items in the batch
        for i, r in enumerate(pb.requests):
            # get mapping
            requests_idx_mapping[r.id] = i

            # setup inputs
            tokenized_inputs = tokenizer(
                r.inputs,
                return_tensors="np",
                padding=padding,
                truncation=truncation,
                return_token_type_ids=False,
                max_length=DEEPSPARSE_SEQUENCE_LENGTH
            )
            input_ids_list.append(tokenized_inputs["input_ids"])
            
            # setup sequence generation helpers, capping at DEEPSPARSE_SEQUENCE_LENGTH
            # cap stopping parameters to DeepSparse sequence length 
            input_len = tokenized_inputs["input_ids"].shape[1]
            assert DEEPSPARSE_SEQUENCE_LENGTH - input_len > 0
            r.stopping_parameters.max_new_tokens = min(
                r.stopping_parameters.max_new_tokens,
                DEEPSPARSE_SEQUENCE_LENGTH - input_len
            )
            stopping_criterias.append(StoppingCriteria.from_pb(r.stopping_parameters, tokenizer))
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))            

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            input_ids_list=input_ids_list,
            past_key_values_list=None
        )

    def __len__(self):
        return len(self.requests)

    def filter(self, request_ids: List[int]) -> Optional["DeepSparseCausalLMBatch"]:
        pass

    def concatenate(cls, batches: List["DeepSparseCausalLMBatch"]) -> "DeepSparseCausalLMBatch":
        pass


class DeepSparseCausalLM:
    def __init__(
            self,
            deployment_path: str
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(deployment_path)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            assert self.tokenizer.eos_token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        

    @property
    def batch_type(self) -> Type[DeepSparseCausalLMBatch]:
        return DeepSparseCausalLMBatch