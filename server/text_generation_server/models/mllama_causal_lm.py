from io import BytesIO
from PIL import Image
import torch
from typing import Iterable
from text_generation_server.pb.generate_pb2 import Request

from dataclasses import dataclass
from opentelemetry import trace
from transformers import (
    PreTrainedTokenizerBase,
)
from typing import Optional

from text_generation_server.models.vlm_causal_lm import VlmCausalLMBatch
from text_generation_server.pb import generate_pb2


tracer = trace.get_tracer(__name__)


@dataclass
class MllamaCausalLMBatch(VlmCausalLMBatch):
    aspect_ratio_ids: Optional[torch.Tensor] = None
    aspect_ratio_mask: Optional[torch.Tensor] = None
    cross_attention_states: Optional[torch.Tensor] = None

    @classmethod
    def batch_tokenized_inputs(
        cls, requests: Iterable[Request], tokenizer, processor, config
    ):
        image_inputs = []
        texts = []
        image_indices = []
        batch_tokenized_inputs = []
        for i, r in enumerate(requests):
            # Each input is encoded into a list, where each element of this input list is either a string or a URL
            curr_text = ""
            has_image = False
            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    curr_text += chunk.text
                elif chunk_type == "image":
                    has_image = True
                    image = Image.open(BytesIO(chunk.image.data))
                    # TODO unsure about BOS
                    curr_text += "<|image|>"
                    image_input = processor.image_processor(image, return_tensors="pt")
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")
            texts.append(curr_text)
            if has_image:
                image_indices.append(i)

            input_ids = tokenizer(
                curr_text,
                truncation=True,
                max_length=r.truncate,
                add_special_tokens=r.add_special_tokens,
            )["input_ids"]
            batch_tokenized_inputs.append(input_ids)
        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat(
                    [img["pixel_values"] for img in image_inputs], dim=0
                ),
            }
            if "aspect_ratio_ids" in image_input:
                new_image_inputs["aspect_ratio_ids"] = torch.cat(
                    [img["aspect_ratio_ids"] for img in image_inputs], dim=0
                )
            if "aspect_ratio_mask" in image_input:
                new_image_inputs["aspect_ratio_mask"] = torch.cat(
                    [img["aspect_ratio_mask"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
            image_inputs["image_indices"] = image_indices
        else:
            image_inputs = None

        return batch_tokenized_inputs, image_inputs

    @classmethod
    def from_pb_processor(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "VlmCausalLMBatch":
        batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(
            pb.requests, tokenizer, processor, config
        )
        batch = cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)
        # XXX: <|image|> token is actually out of bounds and bugs out the logit processors.
        batch.all_input_ids_tensor = batch.all_input_ids_tensor.clamp(
            max=config.text_config.vocab_size - 1
        )
        batch.input_ids = batch.input_ids.clamp(max=config.text_config.vocab_size - 1)

        if image_inputs is not None:
            batch.pixel_values = image_inputs["pixel_values"].to(
                device=device, dtype=dtype
            )
            batch.aspect_ratio_ids = image_inputs["aspect_ratio_ids"].to(device=device)
            batch.aspect_ratio_mask = image_inputs["aspect_ratio_mask"].to(
                device=device
            )
            batch.image_indices = image_inputs["image_indices"]
        else:
            batch.pixel_values = None
            batch.aspect_ratio_ids = None
            batch.aspect_ratio_mask = None
            batch.image_indices = None
        return batch
