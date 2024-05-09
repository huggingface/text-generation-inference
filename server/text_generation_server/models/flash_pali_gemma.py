import torch
import torch.distributed
from opentelemetry import trace
from typing import Optional, Tuple
from text_generation_server.models.vlm_causal_lm import PaliVlmCausalLM
from text_generation_server.models.custom_modeling.flash_pali_gemma_modeling import (
    FlashPaliGemmaForConditionalGeneration,
    PaliGemmaConfig,
)
from transformers import AutoProcessor

tracer = trace.get_tracer(__name__)


class FlashPaliGemma(PaliVlmCausalLM):
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
            # TODO: load in the correct processor based on the model_id
            "google/siglip-base-patch16-224",
            # "google/siglip-so400m-patch14-384",
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        super().__init__(
            config_cls=PaliGemmaConfig,
            model_cls=FlashPaliGemmaForConditionalGeneration,
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            use_medusa=use_medusa,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            prefix="language_model",
        )

    def get_layer_config(self, model) -> Tuple[int, int, int]:
        return (
            len(model.language_model.model.layers),
            model.language_model.model.num_key_value_heads,
            model.language_model.model.head_size,
        )

    def max_past(self) -> Optional[int]:
        return getattr(self.model.language_model, "max_past", None)
