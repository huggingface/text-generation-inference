import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from server.deepsparse.deepsparse_model import (
    DeepSparsePastKeyValues, DeepSparseDecoderModel
)
from server.deepsparse.deepsparse_requests import (
    Request, Batch, CachedBatch, Generation
)

DEEPSPARSE_SEQUENCE_LENGTH = 128
DEEPSPARSE_MULTITOKEN_LENGTH = 4

@dataclass
class DeepSparseCausalLMBatch:    
    batch_id: int
    requests: List[Request]
    requests_idx_mapping: Dict[int,int]
    input_ids_list: List[np.ndarray]
    past_key_values_list: List[Optional[DeepSparsePastKeyValues]]
    
    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        tokenizer: PreTrainedTokenizerBase,
    ) -> "DeepSparseCausalLMBatch":
        
        # parse batch
        requests_idx_mapping = {}
        input_ids_list = []
        
        # setup tokenizer for deepsparse left padding
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        padding, truncation = "longest", False

        # loop through items in the batch
        for idx, r in enumerate(batch.requests):
            requests_idx_mapping[r.id] = idx

            # setup inputs_ids, past_key_values
            tokenized_inputs = tokenizer(
                r.prompt,
                return_tensors="np",
                padding=padding,
                truncation=truncation,
                return_token_type_ids=False,
                max_length=DEEPSPARSE_SEQUENCE_LENGTH
            )
            input_ids_list.append(tokenized_inputs["input_ids"])

        return cls(
            batch_id=batch.id,
            requests=batch.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids_list=input_ids_list,
            past_key_values_list=[None] * len(batch.requests),
        )

    def to_batch(self) -> CachedBatch:
        return CachedBatch(
            batch_id = self.batch_id,
            request_ids=[r.id for r in self.requests],
        )

    # length of the batch
    def __len__(self):
        return len(self.requests)

    # pass list of request ids, returns batch with only those request ids
    def filter(self, request_ids: List[int]) -> Optional["DeepSparseCausalLMBatch"]:
        assert(len(request_ids) > 0)

        requests_idx_mapping = {}
        requests = []
        input_ids_list = []
        past_key_values_list = []

        # loop through requests, keep ones that should remain
        for new_idx, request_id in enumerate(request_ids):
            assert request_id in self.requests_idx_mapping.keys(), "all request ids must be in the batch"
            
            requests_idx_mapping[request_id] = new_idx
            
            old_idx = self.requests_idx_mapping[request_id]
            requests.append(self.requests[old_idx])
            input_ids_list.append(self.input_ids_list[old_idx])
            past_key_values_list.append(self.past_key_values_list[old_idx])

        # update batch state
        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping 
        self.input_ids_list = input_ids_list
        self.past_key_values_list = past_key_values_list

        return self

    # combine two batches into one
    @classmethod
    def concatenate(cls, batches: List["DeepSparseCausalLMBatch"]) -> "DeepSparseCausalLMBatch":
        assert len(batches) > 1, "must have more than 1 batch to concatenate"

        requests_idx_mapping = {}
        requests = []
        input_ids_list = []
        past_key_values_list = []

        start_index = 0
        for i, batch in enumerate(batches):
            assert batch.past_key_values_list is not None, "only concatenate prefilled batches"
            
            # concatenate request, input_ids, and past_key_values lists
            requests.extend(batch.requests)
            input_ids_list.extend(batch.input_ids_list)
            past_key_values_list.extend(batch.past_key_values_list)

            # merge the request_id to index mapping
            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index
            
            start_index += len(batch)

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids_list=input_ids_list,
            past_key_values_list=past_key_values_list
        )

class DeepSparseCausalLM:
    def __init__(
            self,
            model_path: str, 
            tokenizer_path: str,
    ):   
        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            assert self.tokenizer.eos_token
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # setup model
        self.model = DeepSparseDecoderModel(
            onnx_file_path = model_path,
            sequence_length = DEEPSPARSE_SEQUENCE_LENGTH,
            multitoken_length = DEEPSPARSE_MULTITOKEN_LENGTH,
        )

    # TODO (@rsnm2): switch to NextTokenChooser
    def sample_token(
        self,
        logits: np.ndarray
    ):
        assert(logits.shape[0] == 1)        # assert b=1 for now
        return np.argmax(logits[0,-1,:])    # grab logits for the last item in the sequence
    
    # TODO (@rsnm2): switch to StoppingCriteria
    def should_stop(
        self,
        num_tokens_processed: int,
        generated_token_id: int
    ):
        if num_tokens_processed >= self.model.sequence_length:
            return True
        if generated_token_id == self.tokenizer.eos_token_id:
            return True
        return False

    def generate_token(
        self,
        batch: DeepSparseCausalLMBatch,
    ) -> (List[Generation], Optional[DeepSparseCausalLMBatch]):
        
        generations: List[Generation] = []
        all_stopped = True

        # if we supported continuous batching, we would do batched inference here
        # logits, past_key_values = self.model(batch)

        # for each member of the batch:
        #   a) run inference
        #   b) sample and check stopping criteria
        #   c) create generation + update batch
        for i, (
            request,
            input_ids,
            past_key_values,
        ) in enumerate(zip(
            batch.requests, 
            batch.input_ids_list, 
            batch.past_key_values_list
        )):
            
            # run inference
            logits, past_key_values = self.model(input_ids, past_key_values)

            # sample token
            # simple for now --- should use NextTokenChooser
            generated_token_id = self.sample_token(logits)
            
            # check stopping criteria
            # simple for now --- should use StoppingCriteria
            assert len(input_ids.shape) == 2
            assert input_ids.shape[0] == 1
            
            stop = self.should_stop(
                num_tokens_processed=input_ids.shape[1] + 1,
                generated_token_id = generated_token_id
            )
            
            # if not stopped, convert token id to text
            generated_text = None
            if not stop:
                all_stopped = False
                generated_text = self.tokenizer.decode(
                    generated_token_id,
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
            generations.append(Generation(
                request_id=request.id,
                generated_text=generated_text
            ))

            # update values in the batch
            # bad --- this does not occur in place
            assert len(batch.input_ids_list[i].shape) == 2
            assert batch.input_ids_list[i].shape[0] == 1
            batch.input_ids_list[i] = np.append(
                batch.input_ids_list[i],
                np.array([[generated_token_id]]),
                axis=1
            )
            batch.past_key_values_list[i] = past_key_values

        # if all elements of the batch are done, return generation + null for batch
        if all_stopped:
            return generations, None
        
        # return generation + updated batch
        return generations, batch