import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple, List

from text_generation.models import Model


class CausalLM(Model):
    def __init__(self, model_name: str):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()

        super(CausalLM, self).__init__(tokenizer=tokenizer, num_heads=self.model.config.num_attention_heads, device=device)

    def forward(
            self, input_ids, attention_mask, past_key_values: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values
