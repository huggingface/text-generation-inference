import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
from typing import List, Optional, Tuple

from text_generation_server.models import CausalLM


class RW(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        peft_model_id: str = None,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        if peft_model_id:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb_config
            )

            assert 1
            # load Lora weights
            model = PeftModel.from_pretrained(
                model,
                peft_model_id,
                device_map="auto",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                device_map="auto"
                if torch.cuda.is_available() and torch.cuda.device_count() > 1
                else None,
                load_in_8bit=quantize == "bitsandbytes",
                trust_remote_code=trust_remote_code,
            )
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            model = model.cuda()

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        return outputs.logits, outputs.past_key_values
