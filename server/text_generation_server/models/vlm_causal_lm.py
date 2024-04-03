import re

from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict

from text_generation_server.models.flash_mistral import (
    BaseFlashMistral,
    FlashMistralBatch,
)

tracer = trace.get_tracer(__name__)

IMAGES = re.compile(r"!\[[^\]]*\]\((.*?)\s*(\"(?:.*[^\"])\")?\s*\)")


def split(string) -> List[Dict[str, str]]:
    parts = []
    cursor = 0
    for pattern in IMAGES.finditer(string):
        start = pattern.start()
        if start != cursor:
            parts.append({"type": "text", "content": string[cursor:start]})

        parts.append({"type": "image", "content": pattern.group(1)})
        cursor = pattern.end()

    if cursor != len(string):
        parts.append({"type": "text", "content": string[cursor:]})

    return parts


class VlmCausalLMBatch(FlashMistralBatch):
    pass


class VlmCausalLM(BaseFlashMistral):
    @property
    def batch_type(self) -> Type[FlashMistralBatch]:
        return FlashMistralBatch

    def get_layer_config(self, model) -> Tuple[int, int, int]:
        return (
            len(model.language_model.model.layers),
            model.language_model.model.num_key_value_heads,
            model.language_model.model.head_size,
        )

    def max_past(self) -> Optional[int]:
        return getattr(self.model.language_model, "max_past", None)
