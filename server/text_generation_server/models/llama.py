import torch

from transformers import LlamaTokenizer, LlamaForCausalLM
from typing import Optional
from text_generation_server.models import CausalLM


class Llama(CausalLM):
    def __init__(
        self, model_id: str, revision: Optional[str] = None, quantize: bool = False
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = LlamaTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            revision=revision,
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