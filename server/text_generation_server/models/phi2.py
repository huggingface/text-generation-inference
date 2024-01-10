import torch
import torch.nn as nn
import torch.distributed

from typing import Optional, List, Tuple, Type
from text_generation_server.models.types import Generation, Tokens
from text_generation_server.models.causal_lm import CausalLMBatch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase, AutoModelForCausalLM
from text_generation_server.models import CausalLM
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.pb import generate_pb2

class Phi2(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        with device:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                load_in_8bit=quantize == "bitsandbytes",
                trust_remote_code=trust_remote_code
            )
        
        # debug show the model
        print(model)

        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    def decode(self, generated_ids: List[int]) -> str:
        print("🔍 Decoding", generated_ids.shape)
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )


    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ):
        print("🔥 Forwarding", input_ids.shape)
        default = super().forward(input_ids, attention_mask, position_ids, past_key_values)
        return default


    def generate_token(self, batch: CausalLMBatch) -> Tuple[List[Generation], CausalLMBatch | None, Tuple[int, int]]:
        print("🛥️ Generating Tokens")
        default = super().generate_token(batch)
        return default