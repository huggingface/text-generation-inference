import torch
import torch.distributed

from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from text_generation_server.models import CausalLM


class SantaCoder(CausalLM):
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                load_in_8bit=quantize,
                trust_remote_code=True,  # required
            )
            .to(device)
            .eval()
        )

        super(CausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, cleanup_tokenization_spaces=False
        )
