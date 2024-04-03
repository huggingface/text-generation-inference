import torch

from typing import Optional

from transformers import (
    AutoProcessor,
)
from text_generation_server.models.custom_modeling.llava_next import (
    LlavaNextForConditionalGeneration,
)

from text_generation_server.models.vlm_causal_lm import VlmCausalLM


class LlavaNext(VlmCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        super().__init__(
            model_cls=LlavaNextForConditionalGeneration,
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            use_medusa=use_medusa,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
