from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np

from service.model import DeepSparsePastKeyValues, DeepSparseDecoderModel
from utils import Request, Batch, CachedBatch, Generation, StoppingCriteria, NextTokenChooser

DEEPSPARSE_SEQUENCE_LENGTH = 128
DEEPSPARSE_MULTITOKEN_LENGTH = 4

@dataclass
class DeepSparseCausalLMBatch:    
    batch_id: int
    requests: List[Request]
    requests_idx_mapping: Dict[int,int]
    input_ids_list: List[np.ndarray]
    past_key_values_list: List[Optional[DeepSparsePastKeyValues]]
    stopping_criteria_list: List[StoppingCriteria]
    next_token_chooser_list: List[NextTokenChooser]
    
    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        tokenizer: PreTrainedTokenizerBase,
    ) -> "DeepSparseCausalLMBatch":
        
        # parse batch
        requests_idx_mapping = {}
        input_ids_list = []
        stopping_criteria_list = []
        next_token_chooser_list = []

        # loop through items in the batch
        for idx, r in enumerate(batch.requests):
            requests_idx_mapping[r.id] = idx

            # setup inputs_ids, stopping crtieria
            tokenized_inputs = tokenizer(
                r.inputs,
                return_tensors="np",
                padding="longest",
                truncation=False,
                return_token_type_ids=False,
                max_length=DEEPSPARSE_SEQUENCE_LENGTH
            )
            input_ids_list.append(tokenized_inputs["input_ids"])

            # deepsparse able to accept up to seq len tokens
            num_input_tokens = tokenized_inputs["input_ids"].shape[1]
            model_max_new_tokens = DEEPSPARSE_SEQUENCE_LENGTH - num_input_tokens
            stopping_criteria_list.append(
                StoppingCriteria(
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=min(r.generation_parameters.max_new_tokens, model_max_new_tokens)
                )
            )

            # get next token chooser based on input
            next_token_chooser_list.append(
                NextTokenChooser(
                    repetition_penalty=r.generation_parameters.repetition_penalty,
                    temperature=r.generation_parameters.temperature,
                    top_k=r.generation_parameters.top_k,
                    top_p=r.generation_parameters.top_p,
                    do_sample=r.generation_parameters.do_sample,
                    seed=r.generation_parameters.seed,
                )
            )

        return cls(
            batch_id=batch.id,
            requests=batch.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids_list=input_ids_list,
            past_key_values_list=[None] * len(batch.requests),
            stopping_criteria_list=stopping_criteria_list,
            next_token_chooser_list=next_token_chooser_list,
        )

    def to_cached_batch(self) -> CachedBatch:
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

        requests_idx_mapping    = {}
        requests                = []
        input_ids_list          = []
        past_key_values_list    = []
        stopping_criteria_list  = []
        next_token_chooser_list = []

        # loop through requests, keep ones that should remain
        for new_idx, request_id in enumerate(request_ids):
            assert request_id in self.requests_idx_mapping.keys(), "all request ids must be in the batch"
            
            requests_idx_mapping[request_id] = new_idx
            
            old_idx = self.requests_idx_mapping[request_id]
            requests.append(self.requests[old_idx])
            input_ids_list.append(self.input_ids_list[old_idx])
            past_key_values_list.append(self.past_key_values_list[old_idx])
            stopping_criteria_list.append(self.stopping_criteria_list[old_idx])
            next_token_chooser_list.append(self.next_token_chooser_list[old_idx])

        # update batch state
        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping 
        self.input_ids_list = input_ids_list
        self.past_key_values_list = past_key_values_list
        self.stopping_criteria_list = stopping_criteria_list
        self.next_token_chooser_list = next_token_chooser_list

        assert len(self.input_ids_list) == len(self.past_key_values_list)

        return self

    # combine two batches into one
    @classmethod
    def concatenate(cls, batches: List["DeepSparseCausalLMBatch"]) -> "DeepSparseCausalLMBatch":
        assert len(batches) > 1, "must have more than 1 batch to concatenate"

        requests_idx_mapping    = {}
        requests                = []
        input_ids_list          = []
        past_key_values_list    = []
        stopping_criteria_list  = []
        next_token_chooser_list = []

        start_index = 0
        for i, batch in enumerate(batches):
            assert batch.past_key_values_list is not None, "only concatenate prefilled batches"
            
            # concatenate request, input_ids, and past_key_values lists
            requests.extend(batch.requests)
            input_ids_list.extend(batch.input_ids_list)
            #print(f"pkv {past_key_values_list}")
            #print(f"bpkv {batch.past_key_values_list}")
            past_key_values_list.extend(batch.past_key_values_list)
            stopping_criteria_list.extend(batch.stopping_criteria_list)
            next_token_chooser_list.extend(batch.next_token_chooser_list)

            # merge the request_id to index mapping
            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index
            
            start_index += len(batch)

        assert len(input_ids_list) == len(past_key_values_list)

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids_list=input_ids_list,
            past_key_values_list=past_key_values_list,
            stopping_criteria_list=stopping_criteria_list,
            next_token_chooser_list=next_token_chooser_list
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
            batch_size=4
        )

    def generate_token(
        self,
        batch: DeepSparseCausalLMBatch,
    ) -> (List[Generation], Optional[DeepSparseCausalLMBatch]):
        
        generations: List[Generation] = []
        all_stopped = True

        # for each member of the batch:
        #   a) run inference
        #   b) sample and check stopping criteria
        #   c) create generation
        #   d) update batch

        iterator = zip(
            batch.requests, 
            batch.input_ids_list, 
            batch.stopping_criteria_list,
            batch.next_token_chooser_list,
        )

        #assert len(input_ids.shape) == 2
        #assert input_ids.shape[0] == 1

        #print(batch.past_key_values_list)
        #print(len(batch.past_key_values_list))
        #print(batch.input_ids_list)
        #print(len(batch.input_ids_list))

        #print(f"before {len(batch.input_ids_list)} {len(batch.past_key_values_list)}")

        # a) run inference
        logits, batch.past_key_values_list = self.model(batch.input_ids_list, batch.past_key_values_list)

        #print(f"after {len(batch.input_ids_list)} {len(batch.past_key_values_list)} {batch.past_key_values_list}")

        assert len(batch.input_ids_list) == len(batch.past_key_values_list)

        #print(logits)
        #print(logits.shape)

        for i, (
            request, 
            input_ids, 
            stopping_criteria,
            next_token_chooser
        ) in enumerate(iterator):
            # b) sample token and check stopping criteria
            # TODO: should use NextTokenChooser/StoppingCriteria (simple for now)
            generated_token_id = next_token_chooser(input_ids=input_ids, scores=logits[:,-1,:])
            generated_token = self.tokenizer.decode(generated_token_id)
            
            stop, finish_reason = stopping_criteria(generated_token_id=generated_token_id)
            if not stop:
                all_stopped = False
                
            # c) make generation
            generations.append(Generation(
                request_id=request.id,
                token=generated_token,
                token_id=generated_token_id,
                stopped=stop,
                finish_reason=finish_reason
            ))

            # d) update batch 
            # TODO: this does not occur in place
            assert len(batch.input_ids_list[i].shape) == 2
            #assert batch.input_ids_list[i].shape[0] == 1
            batch.input_ids_list[i] = np.append(
                batch.input_ids_list[i],
                np.array([[generated_token_id]]),
                axis=1
            )

        # if all elements of the batch are done, return null for batch
        if all_stopped:
            return generations, None
        
        # return generation + updated batch
        return generations, batch
