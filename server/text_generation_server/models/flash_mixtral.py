import torch

from typing import Optional

from text_generation_server.models.flash_mistral import BaseFlashMistral
from text_generation_server.models.custom_modeling.flash_mixtral_modeling import (
    MixtralConfig,
    FlashMixtralForCausalLM,
)


class FlashMixtral(BaseFlashMistral):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        super(FlashMixtral, self).__init__(
            config_cls=MixtralConfig,
            model_cls=FlashMixtralForCausalLM,
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            use_medusa=use_medusa,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
