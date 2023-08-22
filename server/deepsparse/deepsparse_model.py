import os

os.environ["WAND_OPT_FLAGS"] = "default,~pyramids"

import numpy as np
from typing import Optional, List, Dict

from deepsparse import Context
from deepsparse.engine import LIB
from deepsparse.pipeline import DEEPSPARSE_ENGINE, create_engine
from deepsparse.transformers.utils.helpers import overwrite_onnx_model_inputs, create_causal_mask

PAST_KEY_VALUES_NAME = "past_key_values"

class DeepSparsePastKeyValues:
    def __init__(self):
        prev_num_tokens = 0
        num_frozen_tokens = 1
        self.internal_past_key_values = LIB.kv_cache(prev_num_tokens, num_frozen_tokens)

class DeepSparseDecoderEngine:
    def __init__ (
        self,
        onnx_file_path: str, 
        sequence_length: int = 1024,
        input_ids_length: int = 1,
        engine_context: Optional[Context] = None,
    ):
        
        # setup ONNX graph
        onnx_file_path, cached_outputs, data_type = overwrite_onnx_model_inputs(
            onnx_file_path=onnx_file_path,
            batch_size=1,
            sequence_length=sequence_length,
            input_ids_length=input_ids_length,
        )

        # compile engine
        self.engine = create_engine(
            onnx_file_path=onnx_file_path,
            engine_type=DEEPSPARSE_ENGINE,
            engine_args={"cached_outputs": cached_outputs},
            context=engine_context,
        )
        print(self.engine)

        # save utilties
        self.past_key_value_dtype = data_type
        self.onnx_inputs = self.engine.input_names
        self.empty_past_key_values = self.make_empty_past_key_values()

    # forward function
    def __call__(
        self,
        engine_inputs: Dict[str, np.ndarray],
        past_key_values: DeepSparsePastKeyValues,
        val_inputs: bool = True
    ):
        # format input into lists (we pass empty past key values)
        inputs = [
            self.empty_past_key_values[name] if name.startswith(PAST_KEY_VALUES_NAME) 
            else engine_inputs[name] for name in self.engine.input_names
        ]

        # validate inputs formatted correctly
        if val_inputs:
             self.engine._validate_inputs(inputs)

        # run inference, updates past_key_values internally
        output = self.engine._eng_net.execute_list_out(
            inputs, 
            past_key_values.internal_past_key_values
        )
        logits = output[0]
        return logits, past_key_values

    # empty past kvs (dummy values to be passed around)
    def make_empty_past_key_values(self):
        past_key_values = {}
        for idx, name in enumerate(self.onnx_inputs):
            if name.startswith(PAST_KEY_VALUES_NAME):
                past_key_values[name] = np.zeros(
                    self.engine.input_shapes[idx], 
                    dtype=self.past_key_value_dtype
                )

        return past_key_values
    
class DeepSparseDecoderModel:
    def __init__(
        self,
        onnx_file_path: str,
        sequence_length: int = 1024,
        multitoken_length: int = 16,
        engine_context: Optional[Context] = None,
    ):
        self.sequence_length = sequence_length
        self.multitoken_length = multitoken_length

        # compile decode engine
        self.singletoken_engine = DeepSparseDecoderEngine(
            onnx_file_path=onnx_file_path,
            engine_context=engine_context,
            sequence_length=sequence_length,
            input_ids_length=1,
        )

        # compile prefill engine
        self.multitoken_engine = DeepSparseDecoderEngine(
            onnx_file_path=onnx_file_path,
            engine_context=engine_context,
            sequence_length=sequence_length,
            input_ids_length=self.multitoken_length,
        )

        assert "input_ids" in self.singletoken_engine.onnx_inputs
        assert "attention_mask" in self.singletoken_engine.onnx_inputs
        assert "causal_mask" in self.singletoken_engine.onnx_inputs
        assert "positions" in self.singletoken_engine.onnx_inputs

    def engine_inputs_for_prefill(
        self,
        input_ids: np.ndarray,
    ):
        # split batch into N token_batches
        num_batches = input_ids.shape[1] // self.multitoken_length
        token_batches = [
            input_ids[:, i*self.multitoken_length : (i+1)*self.multitoken_length] 
            for i in range(0, num_batches)
        ]

        # format inputs for each of the N token_batches
        for idx, token_batch in enumerate(token_batches):
            num_processed_tokens = self.multitoken_length * idx

            engine_inputs = {}            
            engine_inputs["input_ids"] = token_batch
            
            # make attention mask from the right
            engine_inputs["attention_mask"] = np.zeros((1, self.sequence_length), dtype=np.int64)
            engine_inputs["attention_mask"][:, -(self.multitoken_length + num_processed_tokens):] = 1
            
            # make positions (building from the right)
            # TODO: handle case when multitoken engine is 1
            assert self.multitoken_length > 1
            engine_inputs["positions"] = np.arange(
                num_processed_tokens, num_processed_tokens + self.multitoken_length
            ).reshape(1, -1).astype(np.int64)

            # make causal mask (building from the right)
            engine_inputs["causal_mask"] = create_causal_mask(
                input_ids=engine_inputs["input_ids"], 
                attention_mask=engine_inputs["attention_mask"]
            )
            yield engine_inputs

    def engine_inputs_for_decode(
        self,
        input_ids: np.ndarray,
    ):
        engine_inputs = {}
        engine_inputs["input_ids"] = input_ids[:,-1:]
        engine_inputs["attention_mask"] = np.zeros((1, self.sequence_length), dtype=np.int64)
        engine_inputs["attention_mask"][:, -input_ids.shape[1]:] = 1

        engine_inputs["causal_mask"] = create_causal_mask(
            engine_inputs["input_ids"],
            engine_inputs["attention_mask"]
        )
        engine_inputs["positions"] = np.array([[input_ids.shape[1] - 1]], dtype=np.int64)
        
        return engine_inputs
    
    def decode(
        self,
        input_ids: np.ndarray,
        past_key_values: DeepSparsePastKeyValues
    ) -> (np.ndarray, DeepSparsePastKeyValues):
        
        # assert input is of shape [1,seq_len] w/ seq_len < self.sequence_len
        assert len(input_ids.shape) == 2
        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] < self.sequence_length
        
        engine_inputs = self.engine_inputs_for_decode(input_ids)
        logits, past_key_values = self.singletoken_engine(
            engine_inputs, 
            past_key_values
        )
        
        return logits, past_key_values
    
    def prefill(
        self,
        input_ids: np.ndarray,
    ) -> (np.ndarray, DeepSparsePastKeyValues):
        
        # assert input is of shape [1,seq_len] w/ seq_len < self.sequence_len
        assert len(input_ids.shape) == 2
        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] < self.sequence_length
        
        tokens_processed = 0
        
        # setup empty past key values
        past_key_values = DeepSparsePastKeyValues()

        # loop through chunks, run inference w/ multitoken engine
        for engine_inputs in self.engine_inputs_for_prefill(input_ids):
            logits, past_key_values = self.multitoken_engine(
                engine_inputs,
                past_key_values
            )
            tokens_processed += self.multitoken_length

        # if anything left over, run inference w/ singletoken engine
        while tokens_processed < input_ids.shape[1]:
            logits, past_key_values = self.decode(
                input_ids=input_ids[:,:tokens_processed+1],
                past_key_values=past_key_values
            )
            tokens_processed += 1
            # print(logits[:,-1:,:])
        
        return logits, past_key_values

    def forward(
        self,
        input_ids: np.ndarray,
        past_key_values: Optional[DeepSparsePastKeyValues] = None,
    ):
        if past_key_values is None:
            return self.prefill(input_ids)
        else:
            return self.decode(input_ids, past_key_values)

    def __call__(
        self,
        input_ids: np.ndarray,
        past_key_values: Optional[DeepSparsePastKeyValues] = None,
    ):
        return self.forward(input_ids, past_key_values)