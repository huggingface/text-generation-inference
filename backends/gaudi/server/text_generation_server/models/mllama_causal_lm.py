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
from text_generation_server.models.flash_causal_lm import (
    prepare_for_decode,
)
from text_generation_server.models.flash_vlm_causal_lm import (
    FlashVlmCausalLMBatch,
    FlashVlmCausalLM,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.layers.attention import (
    Seqlen,
    trim_seqlen_metadata,
    _async_h2d_tensor_copy,
)
import habana_frameworks.torch as htorch
from loguru import logger
from text_generation_server.models.globals import BLOCK_SIZE
from text_generation_server.utils.import_utils import (
    synchronize,
)
import torch.nn.functional as F
from text_generation_server.utils.log import log_master
import time
import os
from vllm_hpu_extension.profiler import HabanaMemoryProfiler, format_bytes

tracer = trace.get_tracer(__name__)


@dataclass
class FlashMllamaCausalLMBatch(FlashVlmCausalLMBatch):
    image_indices: List[int] = 42
    aspect_ratio_ids: Optional[torch.Tensor] = None
    aspect_ratio_mask: Optional[torch.Tensor] = None
    cross_attention_states: Optional[torch.Tensor] = None

    def prepare_for_prefill(
        self, max_padded_input_len, max_padded_bs, max_total_tokens
    ):
        super(FlashVlmCausalLMBatch, self).prepare_for_prefill(
            max_padded_input_len, max_padded_bs, max_total_tokens
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches, padded_total_bs: int = 0):
        batch = super(FlashVlmCausalLMBatch, cls).concatenate(batches, padded_total_bs)
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
        batch = super(FlashVlmCausalLMBatch, self).filter(request_ids)
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
        batch.pixel_values = None
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
    ) -> "FlashVlmCausalLMBatch":
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


def generate_cross_attention_states(
    cross_attention_states, image_indices, input_lengths, pad_seq_len, prefilling
):
    if cross_attention_states is None:
        return None, None, None
    indices_list = []
    if prefilling:
        for i in image_indices:
            indices_list.append(torch.arange(pad_seq_len * i, pad_seq_len * (i + 1)))
        indices = torch.cat(indices_list, dim=0)
    else:
        indices = image_indices[:]
    return indices, input_lengths.index_select(0, image_indices)


class FlashMllamaCausalLM(FlashVlmCausalLM):
    def set_inputs_embeds(self, batch):
        # Set the input embeddings to None, as we are using the input_ids for the model
        batch.inputs_embeds = None

    def warmup_decode(
        self, batch_size: int, block_num: int, batch: FlashMllamaCausalLMBatch
    ):
        input_ids = torch.zeros(batch_size, dtype=batch.input_ids.dtype)
        position_ids = torch.arange(batch_size, dtype=batch.position_ids.dtype)
        blocks = [block_num // batch_size for _ in range(batch_size)]
        blocks[0] += block_num % batch_size
        past_len = []
        block_tables = []
        slots = []
        start_idx = 0

        # fetch the last blocked to warmup block num
        for i in range(batch_size):
            block_array = list(range(start_idx, start_idx + blocks[i]))
            slots.append(BLOCK_SIZE * block_array[-1] + BLOCK_SIZE - 1)
            block_tables.append(block_array)
            past_len.append(blocks[i] * BLOCK_SIZE - 1)
            start_idx += blocks[i]
        input_lengths = torch.ones(batch_size, dtype=torch.int32)

        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )

        hpu_attention_meta = prepare_for_decode(
            self.dtype,
            self.use_contiguous_pa,
            self.device,
            slots,
            block_tables,
            batch_size,
            bucketing_ctx=None,
        )
        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        image_indices = torch.tensor(batch.image_indices)
        image_indices = image_indices.repeat(batch_size)
        cross_attention_states = batch.cross_attention_states.repeat(batch_size, 1, 1)
        indices, cross_attention_len = generate_cross_attention_states(
            cross_attention_states, image_indices, input_lengths, 1, False
        )
        slots_tensor = torch.tensor(slots, dtype=batch.slots.dtype)
        kwargs = {}
        if htorch.utils.internal.is_lazy():
            kwargs["bypass_hpu_graphs"] = not self.use_graphs(
                False, hpu_attention_meta.block_list.shape[0], batch_size
            )
        self.model.forward(
            input_ids=_async_h2d_tensor_copy(input_ids),
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=None,
            kv_cache=self.kv_cache,
            slots=_async_h2d_tensor_copy(slots_tensor),
            seqlen=trim_seqlen_metadata(seqlen),
            hpu_attention_meta=hpu_attention_meta,
            lm_head_indices=None,
            adapter_data=None,
            cross_attention_states=cross_attention_states,
            indices=_async_h2d_tensor_copy(indices),
            cross_attention_len=_async_h2d_tensor_copy(cross_attention_len),
            **kwargs,
        )

    def warmup_prefill(
        self, prompt_len: int, batch_size: int, batch: FlashMllamaCausalLMBatch
    ):
        input_ids = torch.zeros(prompt_len, dtype=batch.input_ids.dtype).repeat(
            batch_size
        )
        position_ids = torch.arange(prompt_len, dtype=batch.position_ids.dtype).repeat(
            batch_size
        )
        max_bt = (prompt_len // BLOCK_SIZE + 1) * batch_size
        block_tables = torch.arange(max_bt, dtype=torch.int32).reshape(batch_size, -1)
        slot_acc = []
        for i in range(batch_size):
            slots = []
            for b in block_tables[i]:
                slots.extend(range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE))
            slot_acc.extend(slots[:prompt_len])
        slots = torch.tensor(slot_acc, dtype=batch.slots.dtype)

        input_lengths = (
            torch.ones(
                batch_size,
                dtype=torch.int32,
            )
            * prompt_len
        )
        cu_seqlen_prefill = torch.zeros(batch_size + 1, dtype=torch.int32)
        torch.cumsum(input_lengths, -1, out=cu_seqlen_prefill[1:])

        lm_head_indices = input_lengths - 1

        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        image_indices = torch.tensor(batch.image_indices)
        image_indices = image_indices.repeat(batch_size)
        cross_attention_states = batch.cross_attention_states.repeat(batch_size, 1, 1)
        indices, cross_attention_len = generate_cross_attention_states(
            cross_attention_states, image_indices, input_lengths, prompt_len, True
        )
        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )
        kwargs = {}
        if htorch.utils.internal.is_lazy():
            kwargs["bypass_hpu_graphs"] = not self.use_graphs(
                True, prompt_len, batch_size
            )
        self.model.forward(
            input_ids=_async_h2d_tensor_copy(input_ids),
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=_async_h2d_tensor_copy(cu_seqlen_prefill),
            kv_cache=self.kv_cache,
            slots=_async_h2d_tensor_copy(slots),
            seqlen=trim_seqlen_metadata(seqlen),
            hpu_attention_meta=None,
            lm_head_indices=_async_h2d_tensor_copy(lm_head_indices),
            adapter_data=None,
            cross_attention_states=cross_attention_states,
            indices=_async_h2d_tensor_copy(indices),
            cross_attention_len=_async_h2d_tensor_copy(cross_attention_len),
            **kwargs,
        )

    def warmup_hpu_graph(self, batch: FlashMllamaCausalLMBatch):
        prompt_graph_mem_ratio = float(os.environ.get("VLLM_GRAPH_PROMPT_RATIO", "0.3"))
        free_mem = HabanaMemoryProfiler.current_free_device_memory()
        graph_free_mem = free_mem - self.mem_reserved
        graph_free_mem = self.align_workers(
            graph_free_mem, torch.distributed.ReduceOp.MIN
        )
        prompt_available_memory = prompt_graph_mem_ratio * graph_free_mem
        decode_available_memory = graph_free_mem - prompt_available_memory
        msg = (
            f"Using {format_bytes(graph_free_mem)}"
            f"/{format_bytes(free_mem)} "
            "of free device memory for HPUGraphs, "
            f"{format_bytes(prompt_available_memory)} for prompt and "
            f"{format_bytes(decode_available_memory)} for decode "
            f"(VLLM_GRAPH_PROMPT_RATIO={prompt_graph_mem_ratio})"
        )
        log_master(logger.info, msg)
        start_time = time.time()
        warmup_shape_count = 0
        warmup_times = 3
        self.bucketing_ctx.generate_prompt_buckets()

        def ordering_function_min_tokens(b):
            return (b[0] * b[1], b[1], b[0])

        buckets = list(
            sorted(self.bucketing_ctx.prompt_buckets, key=ordering_function_min_tokens)
        )
        graph_free_mem
        total_batch_seq = 0.001
        total_mem = 0
        available_mem = prompt_available_memory
        msg = (
            f"Prefill batch size list:{[bsz[0] for bsz in buckets]}\n"
            f"Prefill sequence length list:{[seq[1] for seq in buckets]}\n"
        )
        log_master(logger.info, msg)
        for i, (batch_size, seq_len) in enumerate(buckets):
            if batch_size * seq_len > self.max_batch_prefill_tokens:
                continue
            # Graph memory usage is proportional to seq dimension in a batch
            batch_seq = batch_size * seq_len
            mem_estimate = batch_seq / total_batch_seq * total_mem
            graphed_bucket = (batch_size, seq_len, True)
            if not (
                mem_estimate >= available_mem or batch_seq > self.max_seq_len_to_capture
            ):
                if graphed_bucket not in self.graphed_buckets:
                    self.graphed_buckets.add(graphed_bucket)
            warmup_shape_count += 1
            self.log_warmup(True, i, len(buckets), batch_size, seq_len)
            with HabanaMemoryProfiler() as mem_prof:
                for index in range(warmup_times):
                    self.warmup_prefill(seq_len, batch_size, batch)
                    synchronize(self.device)
            used_mem = self.align_workers(
                mem_prof.consumed_device_memory, torch.distributed.ReduceOp.MAX
            )
            if graphed_bucket in self.graphed_buckets:
                available_mem -= used_mem
                total_mem += used_mem
                total_batch_seq += batch_seq

        log_master(logger.info, "Prefill warmup successful.\n")

        def ordering_function_max_bs(b):
            return (-b[0], b[1])

        self.bucketing_ctx.generate_decode_buckets(self.bucketing_ctx.num_hpu_blocks)
        buckets = list(
            sorted(self.bucketing_ctx.decode_buckets, key=ordering_function_max_bs)
        )
        free_mem = HabanaMemoryProfiler.current_free_device_memory()
        total_batch_seq = 0.001
        total_mem = 0
        available_mem = free_mem - self.mem_reserved
        log_master(
            logger.info, f"Decode batch size list:{[bsz[0] for bsz in buckets]}\n"
        )
        for i, (batch_size, block_num) in enumerate(buckets):
            if batch_size > block_num:
                continue
            # Graph memory usage is proportional to seq dimension in a batch
            batch_seq = batch_size
            mem_estimate = batch_seq / total_batch_seq * total_mem
            graphed_bucket = (batch_size, block_num, False)
            if not mem_estimate >= available_mem:
                if graphed_bucket not in self.graphed_buckets:
                    self.graphed_buckets.add(graphed_bucket)
            warmup_shape_count += 1
            self.log_warmup(False, i, len(buckets), batch_size, block_num)
            with HabanaMemoryProfiler() as mem_prof:
                for index in range(warmup_times):
                    self.warmup_decode(batch_size, block_num, batch)
                    synchronize(self.device)
            used_mem = self.align_workers(
                mem_prof.consumed_device_memory, torch.distributed.ReduceOp.MAX
            )
            if graphed_bucket in self.graphed_buckets:
                available_mem -= used_mem
                total_mem += used_mem
                total_batch_seq += batch_seq

        log_master(logger.info, "Decode warmup successful.\n")

        log_master(
            logger.info,
            f"warmup hpu graph time {int(time.time() - start_time)}s warmup shape count {warmup_shape_count}",
        )

    def forward(
        self,
        batch: FlashMllamaCausalLMBatch,
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
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        if batch.pixel_values is not None:
            cross_attention_states = self.model.vision_forward(
                pixel_values=batch.pixel_values,
                aspect_ratio_ids=batch.aspect_ratio_ids,
                aspect_ratio_mask=batch.aspect_ratio_mask,
            )
            batch.cross_attention_states = cross_attention_states

        cross_attention_states = batch.cross_attention_states

        kwargs = {}
        if htorch.utils.internal.is_lazy():
            batch_size = input_lengths.shape[0]
            seqlen = (
                input_ids.shape[0] // batch_size
                if batch.prefilling
                else batch.hpu_attn_meta.block_list.shape[0]
            )
            kwargs["bypass_hpu_graphs"] = not self.use_graphs(
                batch.prefilling, seqlen, batch_size
            )

        if batch.prefill_cache_indices is not None:
            slots_pad = torch.zeros_like(input_ids, device=slots.device)
            slots_pad[batch.prefill_cache_indices] = slots
            slots = slots_pad
        else:
            slots_pad = torch.zeros_like(input_ids, device=slots.device)
            slots_pad[: slots.shape[0]] = slots
            slots = slots_pad
        orig_bs = len(batch)
        padded_bs = batch.input_lengths_tensor.shape[0]
        padded_input_len = input_ids.view(padded_bs, -1).shape[-1]
        image_indices = torch.tensor(batch.image_indices)

        if cross_attention_states is not None:
            cross_attention_states = F.pad(
                cross_attention_states,
                (0, 0, 0, 0, 0, (padded_bs - orig_bs)),
                value=0,
            )
        if len(image_indices) != 0:
            pad_indices = torch.arange(orig_bs, padded_bs)
            image_indices = torch.cat((image_indices, pad_indices), dim=0)

        indices, cross_attention_len = generate_cross_attention_states(
            cross_attention_states,
            image_indices,
            input_lengths,
            padded_input_len,
            batch.prefilling,
        )
        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )
        logits, speculative_logits = self.model.forward(
            input_ids=input_ids,
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=_async_h2d_tensor_copy(cu_seqlen_prefill),
            kv_cache=kv_cache,
            slots=_async_h2d_tensor_copy(slots),
            seqlen=trim_seqlen_metadata(seqlen),
            hpu_attention_meta=batch.hpu_attn_meta,
            lm_head_indices=_async_h2d_tensor_copy(lm_head_indices),
            # TODO list
            adapter_data=None,
            cross_attention_states=cross_attention_states,
            indices=_async_h2d_tensor_copy(indices),
            cross_attention_len=_async_h2d_tensor_copy(cross_attention_len),
            **kwargs,
        )
        if batch.pixel_values is not None:
            batch.pixel_values = None
        return logits, speculative_logits
