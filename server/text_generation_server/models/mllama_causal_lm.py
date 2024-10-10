import torch

import numpy as np

from typing import Iterable, Optional, Tuple, List, Dict
from text_generation_server.pb.generate_pb2 import Request
from io import BytesIO
from PIL import Image
from dataclasses import dataclass
from opentelemetry import trace
from transformers import (
    PreTrainedTokenizerBase,
)

from text_generation_server.models.vlm_causal_lm import VlmCausalLMBatch, VlmCausalLM
from text_generation_server.pb import generate_pb2
from text_generation_server.models.flash_causal_lm import (
    block_tables_to_ragged,
)
from text_generation_server.models.globals import PREFIX_CACHING, ATTENTION
from text_generation_server.layers.attention import Seqlen


tracer = trace.get_tracer(__name__)


@dataclass
class MllamaCausalLMBatch(VlmCausalLMBatch):
    image_indices: List[int] = 42
    aspect_ratio_ids: Optional[torch.Tensor] = None
    aspect_ratio_mask: Optional[torch.Tensor] = None
    cross_attention_states: Optional[torch.Tensor] = None

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches):
        batch = super().concatenate(batches)
        batch.pixel_values = None
        batch.pixel_attention_mask = None

        offset = 0
        image_indices = []
        attention_states = []
        for b in batches:
            if b.cross_attention_states is not None:
                attention_states.append(b.cross_attention_states)
            image_indices.extend([i + offset for i in b.image_indices])
            offset += len(b.image_indices)
        if len(attention_states) > 0:
            assert len(image_indices) > 0
            batch.cross_attention_states = torch.cat(attention_states, dim=0)
            batch.image_indices = image_indices
        else:
            batch.cross_attention_states = None
            batch.image_indices = []
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        assert self.image_indices is not None
        batch = super().filter(request_ids)
        assert self.image_indices is not None
        indices = []
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)

        offset = 0
        new_image_indices = []
        prev_i = None
        for i in self.image_indices:
            if i in indices:
                new_image_indices.append(offset)
                if i != prev_i:
                    offset += 1
                prev_i = i

        batch.image_indices = new_image_indices
        if len(new_image_indices) > 0:
            assert max(new_image_indices) < self.cross_attention_states.shape[0]
            assert offset <= self.cross_attention_states.shape[0]
            batch.cross_attention_states = self.cross_attention_states[
                new_image_indices
            ]
        else:
            batch.cross_attention_states = None
        return batch

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
            curr_image = None
            curr_i = None
            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    curr_text += chunk.text
                elif chunk_type == "image":
                    image = Image.open(BytesIO(chunk.image.data))
                    # TODO unsure about BOS
                    curr_text += "<|image|>"
                    image_input = processor.image_processor(image, return_tensors="pt")
                    curr_image = image_input
                    curr_i = i
                    # image_inputs.append(image_input)
                    # image_indices.append(i)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")
            texts.append(curr_text)
            if curr_image is not None:
                image_inputs.append(curr_image)
                image_indices.append(curr_i)

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

        if image_inputs is not None:
            assert len(image_indices) == image_inputs["pixel_values"].shape[0]

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
        if isinstance(batch.input_ids, list):
            if len(batch) > 1:
                input_ids = np.concatenate(batch.input_ids, dtype=np.int64)
            else:
                input_ids = batch.input_ids[0]
            batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)

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
            batch.image_indices = []
        assert batch.image_indices is not None
        return batch


class MllamaCausalLM(VlmCausalLM):
    def forward(
        self,
        batch: MllamaCausalLMBatch,
        adapter_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Model Forward
        if batch.speculative_ids is not None:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = self.kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

            speculative_ids = batch.speculative_ids

            B, speculative_length = speculative_ids.shape
            new_length = speculative_length + 1
            new_input_ids = torch.cat(
                [input_ids.unsqueeze(-1), speculative_ids], dim=1
            ).reshape(-1)
            arange = torch.arange(new_length, device=position_ids.device).unsqueeze(0)
            arange_int = arange.to(dtype=torch.int32)
            new_position_ids = (
                position_ids.unsqueeze(-1).expand(B, new_length) + arange
            ).view(-1)
            slots = (slots.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            input_lengths = (
                input_lengths.unsqueeze(-1).expand(B, new_length) + arange_int
            ).view(-1)
            cache_lengths_tensor = (
                batch.cache_lengths_tensor.unsqueeze(-1).expand(B, new_length)
            ).reshape(-1)

            # Add Copy the block tables for all members
            block_tables = (
                block_tables.unsqueeze(1)
                .expand(B, new_length, -1)
                .reshape(B * new_length, -1)
                .contiguous()
            )
            max_s = max_s + speculative_length

            input_ids = new_input_ids
            position_ids = new_position_ids
        else:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = self.kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            cache_lengths_tensor = batch.cache_lengths_tensor
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        # Try to find an associated cuda graph
        bs = input_ids.shape[0]
        sorted_padded_bs = sorted([k for k in self.cuda_graphs.keys() if k >= bs])
        if sorted_padded_bs:
            # Get associated cuda graph
            cuda_graph = self.cuda_graphs[sorted_padded_bs[0]]
        else:
            cuda_graph = None
        if (
            cu_seqlen_prefill is not None
            or cuda_graph is None
            # Only run cuda graphs when there's no images.
            or batch.cross_attention_states is not None
        ):
            if PREFIX_CACHING:
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=batch.input_lengths,
                    cache_lengths=batch.cache_lengths,
                )
            with self._forward_context(
                block_tables=block_tables,
                cu_seqlen_prefill=cu_seqlen_prefill,
                input_lengths_tensor=input_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
            ):
                seqlen = Seqlen(
                    input_lengths=input_lengths,
                    cache_lengths=cache_lengths_tensor,
                    cu_seqlen_q=cu_seqlen_prefill,
                    max_q=batch.max_input_length,
                    max_k=batch.max_current_length,
                )

                if batch.pixel_values is not None:
                    cross_attention_states = self.model.vision_forward(
                        pixel_values=batch.pixel_values,
                        aspect_ratio_ids=batch.aspect_ratio_ids,
                        aspect_ratio_mask=batch.aspect_ratio_mask,
                    )
                    batch.cross_attention_states = cross_attention_states

                cross_attention_states = batch.cross_attention_states

                logits, speculative_logits = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seqlen_prefill=cu_seqlen_prefill,
                    kv_cache=kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    seqlen=seqlen,
                    max_s=max_s,
                    prefill_cache_indices=batch.prefill_cache_indices,
                    lm_head_indices=lm_head_indices,
                    cross_attention_states=cross_attention_states,
                    adapter_data=adapter_data,
                    image_indices=batch.image_indices[:],
                )
                if batch.prefill_cache_indices is not None:
                    batch.prefill_cache_indices = None
                if batch.pixel_values is not None:
                    batch.pixel_values = None
                return logits, speculative_logits

        # Copy inputs to the static inputs of the cuda graph
        # Static inputs are potentially padded
        cuda_graph["input_ids"][: input_ids.shape[0]] = input_ids
        cuda_graph["position_ids"][: position_ids.shape[0]] = position_ids
        if ATTENTION == "flashinfer":
            block_tables = block_tables_to_ragged(
                block_tables=block_tables,
                input_lengths=batch.input_lengths,
                cache_lengths=batch.cache_lengths,
            )
            cuda_graph["block_tables"][: block_tables.shape[0]] = block_tables
        else:
            cuda_graph["block_tables"][
                : block_tables.shape[0], : block_tables.shape[1]
            ] = block_tables

        # XXX: This is working only because block 0 is reserved for the healthcheck
        # so it doesn't matter if we override it with bogus values.
        cuda_graph["slots"].fill_(0)
        cuda_graph["slots"][: slots.shape[0]] = slots
        cuda_graph["input_lengths"].zero_()
        cuda_graph["input_lengths"][: input_lengths.shape[0]] = input_lengths
        cuda_graph["cache_lengths"].zero_()
        cuda_graph["cache_lengths"][
            : cache_lengths_tensor.shape[0]
        ] = cache_lengths_tensor

        with self._forward_context(
            block_tables=cuda_graph["block_tables"],
            cu_seqlen_prefill=None,
            input_lengths_tensor=cuda_graph["input_lengths"],
            cache_lengths_tensor=cuda_graph["cache_lengths"],
            state=cuda_graph["state"],
        ):
            # Replay the graph
            cuda_graph["graph"].replay()

        # Slice output to the correct shape
        speculative_logits = (
            cuda_graph["speculative_logits"][:bs]
            if cuda_graph["speculative_logits"] is not None
            else None
        )
        logits = cuda_graph["logits"][:bs]
        return logits, speculative_logits
