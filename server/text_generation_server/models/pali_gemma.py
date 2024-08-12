import torch
import torch.distributed
from opentelemetry import trace
from typing import Optional, Tuple
from text_generation_server.models.vlm_causal_lm import (
    VlmCausalLM,
    VlmCausalLMBatch,
    image_text_replacement,
    load_data_uri,
    split,
)
from text_generation_server.models.custom_modeling.flash_pali_gemma_modeling import (
    PaliGemmaForConditionalGeneration,
)
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor

tracer = trace.get_tracer(__name__)


class PaliGemmaBatch(VlmCausalLMBatch):
    @classmethod
    def batch_tokenized_inputs(cls, requests, tokenizer, processor, config):
        batch_inputs = []
        image_inputs = []
        max_truncation = 0
        for r in requests:
            chunks = split(r.inputs)
            full_text = ""
            image_id = 0
            for chunk in chunks:
                if chunk["type"] == "text":
                    full_text += "<bos>" + chunk["content"] + "\n"
                elif chunk["type"] == "image":
                    image = chunk["content"]
                    # Should never receive URLs anymore, processing should be done
                    # On the rust layer.
                    # This avoid making n queries per TP
                    # if image.startswith("https://") or image.startswith("http://"):
                    #     image = processor.image_processor.fetch_images(image)
                    if image.startswith("data:"):
                        image = load_data_uri(image)
                    else:
                        raise RuntimeError(
                            "Cannot process input image not starting with data:"
                        )
                    # TODO do_convert_RGB should be on by default ?
                    image = image.convert("RGB")
                    image_input = processor.image_processor(image, return_tensors="pt")
                    full_text += image_text_replacement(image_input, config, image_id)
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk['type']}")

            batch_inputs.append(full_text)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs,
            truncation=True,
            max_length=max_truncation,
            add_special_tokens=False,
        )["input_ids"]
        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat(
                    [img["pixel_values"] for img in image_inputs], dim=0
                ),
            }
            if "pixel_attention_mask" in image_input:
                new_image_inputs["pixel_attention_mask"] = torch.cat(
                    [img["pixel_attention_mask"] for img in image_inputs], dim=0
                )
            if "image_sizes" in image_input:
                new_image_inputs["image_sizes"] = torch.cat(
                    [img["image_sizes"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
        else:
            image_inputs = None
        return batch_tokenized_inputs, image_inputs


class PaliGemma(VlmCausalLM):
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
        )

        super().__init__(
            config_cls=AutoConfig,
            model_cls=PaliGemmaForConditionalGeneration,
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @property
    def batch_type(self):
        return PaliGemmaBatch

    def get_layer_config(self, model) -> Tuple[int, int, int]:
        return (
            len(model.text_model.model.layers),
            model.text_model.model.num_key_value_heads,
            model.text_model.model.head_size,
        )

    def max_past(self) -> Optional[int]:
        return getattr(self.model.text_model, "max_past", None)
