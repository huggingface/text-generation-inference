import torch
import torch.distributed

from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from text_generation.models import CausalLM

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"


class SantaCoder(CausalLM):
    def __init__(self, model_name: str, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    EOD,
                    FIM_PREFIX,
                    FIM_MIDDLE,
                    FIM_SUFFIX,
                    FIM_PAD,
                ],
                "pad_token": EOD,
            }
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize,
            trust_remote_code=True,  # required
        ).eval()

        super(CausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, cleanup_tokenization_spaces=False
        )

    def forward(
        self, input_ids, attention_mask, past_key_values: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # FIXME: current forward with past is bugged for bigcode/santacoder because past_key_values does not have
        #        the correct shape ([batch_size, D, seq_length] instead of [batch_size, seq_length D]
        #        this leads to position_ids being wrong

        input_length = input_ids.shape[-1]
        past_key_values_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[-1]
        )
        position_ids = torch.arange(
            past_key_values_length,
            input_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        ).view(1, input_length)

        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values
