import torch

from typing import Optional, Tuple

from transformers import (
    AutoProcessor,
)
from text_generation_server.models.custom_modeling.idefics2 import (
    Idefics2ForConditionalGeneration,
)

from text_generation_server.models.vlm_causal_lm import VlmCausalLM


class Idefics2(VlmCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # XXX: Extremely important to cap resolution in order to limit
            # VRAM usage.
            size={"longest_edge": 448, "shortest_edge": 378},
        )
        super().__init__(
            model_cls=Idefics2ForConditionalGeneration,
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    def get_layer_config(self, model) -> Tuple[int, int, int]:
        return (
            len(model.text_model.model.layers),
            model.text_model.model.num_key_value_heads,
            model.text_model.model.head_size,
        )

    def max_past(self) -> Optional[int]:
        return getattr(self.model.text_model, "max_past", None)
