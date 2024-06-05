import math
import os
import time
import itertools
import torch
import torch.distributed

import numpy as np

from loguru import logger
from dataclasses import dataclass
from opentelemetry import trace
from transformers import PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models import Model
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.utils.dist import RANK
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.models.globals import MEM_POOL, CUDA_GRAPHS
import text_generation_server.models.globals as tgi_globals
from text_generation_server.utils import StoppingCriteria, HeterogeneousNextTokenChooser
from text_generation_server.utils.dist import MEMORY_FRACTION

from text_generation_server.utils.import_utils import (
    empty_cache,
    synchronize,
    get_free_memory,
)

tracer = trace.get_tracer(__name__)

BLOCK_SIZE: int = 16

# Will be set in init
SLIDING_WINDOW: Optional[int] = None


def set_sliding_window(sliding_window: int):
    global SLIDING_WINDOW
    SLIDING_WINDOW = sliding_window


def get_sliding_windows() -> int:
    global SLIDING_WINDOW
    return SLIDING_WINDOW


@dataclass
class FlashCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    # request id -> idx in list mapping
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    speculative_ids: Optional[torch.Tensor]

    # Flash Attention values

    # tensor of length b containing the cumulative sequence lengths of the sequences in the batch, only used in prefill
    cu_seqlen_prefill: Optional[torch.Tensor]
    # Prefill cache indices is used to slice into the kv tensor before caching it into the paged attention buffers
    # as we only keep SLIDING_WINDOW values instead of the whole tensor
    prefill_cache_indices: Optional[torch.Tensor]

    # Paged Attention values

    # Set when creating the batch
    # CPU tensor of length b indicating the start of each sequence in slots
    start_slots: torch.Tensor
    # tensor of indices of the currently used slots, length = \sum_{i=0}^{b} s_i in prefill, length = b in decode
    slot_indices: torch.Tensor

    # list of length b of list of length s_i // block_size
    block_tables: List[List[int]]
    # tensor of size [b, max_total_seqlen // block_size] holding the paged attention block tables for all sequences
    block_tables_tensor: torch.Tensor
    # tensor of length \sum_{i=0}^{b} max_s_i  holding the paged attention slots for all sequences
    slots: torch.Tensor

    max_seqlen: int

    # Prefill metadata tensors to efficiently compute logprobs
    prefill_head_indices: Optional[torch.Tensor]
    prefill_next_token_indices: Optional[torch.tensor]
    prefill_cu_outlens: Optional[List[int]]

    # All tokens
    all_input_ids: List[List[int]]
    all_input_ids_tensor: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    input_lengths_tensor: torch.Tensor
    prefix_offsets: List[Optional[int]]
    read_offsets: List[Optional[int]]

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    stopping_criterias: List[StoppingCriteria]
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    # Number of blocks in this batch
    num_blocks: int
    # Maximum number of blocks
    max_blocks: int

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.num_blocks * BLOCK_SIZE,
        )

    @classmethod
    def batch_tokenized_inputs(cls, requests, tokenizer):
        batch_inputs = []
        max_truncation = 0
        for r in requests:
            batch_inputs.append(r.inputs)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_truncation
        )["input_ids"]
        return batch_tokenized_inputs

    @classmethod
    def from_tokenized(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        batch_tokenized_inputs,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        sliding_window = get_sliding_windows()
        position_ids = []
        cu_seqlen_prefill = [0]
        start_slots = []
        slot_indices = []
        prefill_cache_indices = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        requests_idx_mapping = {}

        all_prefill_logprobs = True
        no_prefill_logprobs = True
        prefill_head_indices = []
        prefill_next_token_indices = []
        prefill_cu_outlens = [0]

        next_token_chooser_parameters = []
        stopping_criterias = []
        top_n_tokens = []

        # Cumulative length
        cumulative_length = 0
        cumulative_max_length = 0
        prefill_out_cumulative_length = 0

        num_blocks = 0
        max_seqlen = 0
        max_length = 0
        max_blocks = 0

        block_tables = []
        slots = []

        # Parse batch
        for i, (r, tokenized_input) in enumerate(
            zip(pb.requests, batch_tokenized_inputs)
        ):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenized_input[-r.truncate :]
            if (
                tokenized_input[0] == tokenizer.bos_token_id
                and tokenized_input[1] == tokenizer.bos_token_id
            ):
                tokenized_input = tokenized_input[1:]

            input_length = len(tokenized_input)
            input_lengths.append(input_length)

            prefix_offsets.append(input_length - 5)
            read_offsets.append(input_length)

            all_input_ids.append(tokenized_input)

            # Position ids
            request_position_ids = torch.arange(0, input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            # Add cumulative lengths of all previous inputs
            cu_seqlen_prefill.append(cumulative_length + input_length)

            next_token_chooser_parameters.append(r.parameters)

            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            max_new_tokens = stopping_criteria.max_new_tokens
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)

            # Paged attention
            # Remove one as the first token des not have a past
            speculative_length = get_speculate()
            speculative_length = 0 if speculative_length is None else speculative_length
            total_tokens = input_length + max_new_tokens - 1 + speculative_length

            # blocks and slots can be empty (for example in warmup)
            if not r.blocks:
                needed_blocks = math.ceil(total_tokens / BLOCK_SIZE)
                request_blocks = [
                    b for b in range(num_blocks, num_blocks + needed_blocks)
                ]
                request_slots = [
                    s
                    for b in request_blocks
                    for s in range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
                ]
            else:
                request_blocks = r.blocks
                request_slots = r.slots

            block_tables.append(request_blocks)
            slots.extend(request_slots[:total_tokens])
            num_blocks += len(request_blocks)
            start_slots.append(cumulative_max_length)

            request_slot_indices = torch.arange(
                cumulative_max_length,
                cumulative_max_length + input_length,
                dtype=torch.int64,
            )
            slot_indices.append(request_slot_indices)

            # Create tensor to slice into the kv tensor in prefill
            if sliding_window is not None:
                request_prefill_cache_indices = torch.arange(
                    cumulative_length + max(0, input_length - sliding_window),
                    cumulative_length + input_length,
                    dtype=torch.int64,
                )
                prefill_cache_indices.append(request_prefill_cache_indices)

            all_prefill_logprobs = all_prefill_logprobs and r.prefill_logprobs
            no_prefill_logprobs = no_prefill_logprobs and not r.prefill_logprobs

            if r.prefill_logprobs:
                prefill_head_indices.append(request_position_ids + cumulative_length)
                prefill_next_token_indices.append(
                    prefill_out_cumulative_length + input_length - 1
                )
                prefill_cu_outlens.append(prefill_out_cumulative_length + input_length)
                prefill_out_cumulative_length += input_length
            else:
                prefill_head_indices.append(
                    torch.tensor(
                        [cumulative_length + input_length - 1], dtype=torch.int32
                    )
                )
                prefill_next_token_indices.append(prefill_out_cumulative_length)
                prefill_cu_outlens.append(prefill_out_cumulative_length + 1)
                prefill_out_cumulative_length += 1

            # Update
            cumulative_length += input_length
            cumulative_max_length += total_tokens
            max_seqlen = max(max_seqlen, input_length)
            max_blocks = max(max_blocks, len(request_blocks))
            max_length = max(
                max_length, input_length + max_new_tokens + speculative_length
            )

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype, device, tokenizer
        )
        start_slots = torch.tensor(start_slots, dtype=torch.int64)

        # Padded all_input_ids_tensor
        all_input_ids_tensor = np.zeros(
            (len(all_input_ids), max_length), dtype=np.int64
        )
        for i, input_ids in enumerate(all_input_ids):
            all_input_ids_tensor[i, : len(input_ids)] = input_ids

        # Create tensors on device
        all_input_ids_tensor = torch.tensor(
            all_input_ids_tensor, dtype=torch.int64, device=device
        )

        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)
            slot_indices = torch.cat(slot_indices)
            if sliding_window is not None:
                prefill_cache_indices = torch.cat(prefill_cache_indices)
        else:
            input_ids = all_input_ids[0]
            position_ids = position_ids[0]
            slot_indices = slot_indices[0]
            if sliding_window is not None:
                prefill_cache_indices = prefill_cache_indices[0]

        cu_seqlen_prefill = torch.tensor(
            cu_seqlen_prefill, device=device, dtype=torch.int32
        )
        position_ids = position_ids.to(device)
        slot_indices = slot_indices.to(device)
        prefill_cache_indices = (
            prefill_cache_indices.to(device) if sliding_window is not None else None
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        input_lengths_tensor = torch.tensor(
            input_lengths, dtype=torch.int32, device=device
        )

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = cu_seqlen_prefill[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = cu_seqlen_prefill[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.tensor(
                torch.cat(prefill_head_indices), dtype=torch.int64, device=device
            )
            prefill_next_token_indices = torch.tensor(
                prefill_next_token_indices, dtype=torch.int64, device=device
            )
        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        slots = torch.tensor(slots, dtype=torch.int64, device=device)
        block_tables_tensor = torch.zeros(
            (len(block_tables), max_blocks), dtype=torch.int32, device="cpu"
        )
        for i, request_blocks in enumerate(block_tables):
            block_tables_tensor[i, : len(request_blocks)] = torch.tensor(request_blocks)
        block_tables_tensor = block_tables_tensor.to(device)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            prefill_cache_indices=prefill_cache_indices,
            start_slots=start_slots,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            slots=slots,
            max_seqlen=max_seqlen,
            prefill_head_indices=prefill_head_indices,
            prefill_next_token_indices=prefill_next_token_indices,
            prefill_cu_outlens=prefill_cu_outlens,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            speculative_ids=None,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        batch_tokenized_inputs = cls.batch_tokenized_inputs(pb.requests, tokenizer)
        return cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> "FlashCausalLMBatch":
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        # We assume that if len(requests) == len(self) then the requests are the same
        if len(request_ids) == len(self):
            return self

        device = self.input_ids.device

        # New values after filtering
        requests_idx_mapping = {}

        # Used to index into tensors
        indices = []

        # slots to keep after filtering
        slot_filtering_indices = torch.zeros(
            self.slots.shape[0], dtype=torch.bool, device=device
        )

        # Create on CPU to only move to GPU once instead of at every copy
        slot_indices = torch.empty(len(request_ids), dtype=torch.int64)
        max_seqlen = 0

        requests = []
        start_slots = []
        block_tables = []
        all_input_ids = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        stopping_criterias = []
        top_n_tokens = []

        num_blocks = 0
        max_blocks = 0
        # Cumulative length
        cumulative_max_length = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
            requests_idx_mapping[request_id] = i

            requests.append(self.requests[idx])

            # Get length
            request_input_length = self.input_lengths[idx]
            max_seqlen = max(max_seqlen, request_input_length)

            all_input_ids.append(self.all_input_ids[idx])

            input_lengths.append(request_input_length)
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)

            top_n_tokens.append(self.top_n_tokens[idx])

            remaining_tokens = (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )

            request_block_table = self.block_tables[idx]
            num_blocks += len(request_block_table)
            block_tables.append(request_block_table)
            start_slots.append(cumulative_max_length)

            # Copy to tensor (CPU)
            slot_indices[i] = cumulative_max_length + request_input_length - 1

            # Set slice
            slot_filtering_indices[
                self.start_slots[idx] : self.start_slots[idx]
                + request_input_length
                + remaining_tokens
                - 1
            ] = True

            cumulative_max_length += request_input_length + remaining_tokens - 1

            max_blocks = max(max_blocks, len(request_block_table))

        # Index into tensors
        input_ids = self.input_ids[indices]
        position_ids = self.position_ids[indices]
        all_input_ids_tensor = self.all_input_ids_tensor[indices]
        block_tables_tensor = self.block_tables_tensor[indices]
        input_lengths_tensor = self.input_lengths_tensor[indices]
        slots = self.slots[slot_filtering_indices]
        next_token_chooser = self.next_token_chooser.filter(indices)
        top_n_tokens_tensor = self.top_n_tokens_tensor[indices]
        speculative_ids = (
            self.speculative_ids[indices] if self.speculative_ids is not None else None
        )

        start_slots = torch.tensor(start_slots, dtype=torch.int64)

        # Move to GPU now that we have the whole tensor
        slot_indices = slot_indices.to(device)

        return type(self)(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            start_slots=start_slots,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            slots=slots,
            max_seqlen=max_seqlen,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            speculative_ids=speculative_ids,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        requests_idx_mapping = {}

        num_blocks = 0
        total_batch_size = 0
        total_slots = 0
        max_blocks = 0
        max_length = 0
        max_seqlen = 0
        for b in batches:
            total_batch_size += len(b)
            total_slots += len(b.slots)
            num_blocks += b.num_blocks
            speculative_length = (
                b.speculative_ids.shape[1] if b.speculative_ids is not None else 0
            )
            max_blocks = max(max_blocks, b.max_blocks)
            max_seqlen = max(max_seqlen, b.max_seqlen)
            max_length = max(
                max_length,
                max(
                    input_length
                    + stopping_criteria.max_new_tokens
                    + speculative_length
                    - stopping_criteria.current_tokens
                    for input_length, stopping_criteria in zip(
                        b.input_lengths, b.stopping_criterias
                    )
                ),
            )

        input_ids = batches[0].input_ids.new_empty(total_batch_size)
        position_ids = batches[0].position_ids.new_empty(total_batch_size)
        slots = batches[0].slots.new_empty(total_slots)
        slot_indices = batches[0].slot_indices.new_empty(total_batch_size)
        input_lengths_tensor = batches[0].input_lengths_tensor.new_empty(
            total_batch_size
        )
        block_tables_tensor = batches[0].block_tables_tensor.new_zeros(
            (total_batch_size, max_blocks)
        )
        all_input_ids_tensor = batches[0].all_input_ids_tensor.new_zeros(
            (total_batch_size, max_length)
        )
        top_n_tokens_tensor = batches[0].top_n_tokens_tensor.new_zeros(
            total_batch_size,
        )

        start_slots = []
        block_tables = []
        all_input_ids = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        next_token_chooser_parameters = []
        fsm_grammar_states = []
        stopping_criterias = []
        top_n_tokens = []

        # Cumulative length
        cumulative_batch_size = 0
        cumulative_slots = 0

        for i, batch in enumerate(batches):
            requests.extend(batch.requests)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + cumulative_batch_size

            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + len(batch)
            slots_start_index = cumulative_slots
            slots_end_index = cumulative_slots + len(batch.slots)

            # Copy tensors (GPU)
            input_ids[start_index:end_index] = batch.input_ids
            position_ids[start_index:end_index] = batch.position_ids
            slot_indices[start_index:end_index] = batch.slot_indices + cumulative_slots
            input_lengths_tensor[start_index:end_index] = batch.input_lengths_tensor
            top_n_tokens_tensor[start_index:end_index] = batch.top_n_tokens_tensor
            slots[slots_start_index:slots_end_index] = batch.slots

            all_input_ids_tensor[
                start_index:end_index, : batch.all_input_ids_tensor.shape[1]
            ] = batch.all_input_ids_tensor[:, :max_length]

            block_tables_tensor[
                start_index:end_index, : batch.block_tables_tensor.shape[1]
            ] = batch.block_tables_tensor[:, :max_blocks]

            start_slots.append(batch.start_slots + cumulative_slots)

            block_tables.extend(batch.block_tables)
            all_input_ids.extend(batch.all_input_ids)

            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)

            next_token_chooser_parameters.extend([r.parameters for r in batch.requests])
            fsm_grammar_states.extend(batch.next_token_chooser.fsm_grammar_states)
            stopping_criterias.extend(batch.stopping_criterias)

            top_n_tokens.extend(batch.top_n_tokens)

            # Update
            cumulative_batch_size += len(batch)
            cumulative_slots += len(batch.slots)

        start_slots = torch.concat(start_slots)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters,
            dtype=batches[0].next_token_chooser.dtype,
            device=batches[0].next_token_chooser.device,
            tokenizer=batches[0].next_token_chooser.tokenizer,
            fsm_grammar_states=fsm_grammar_states,
        )

        speculative_ids = (
            torch.cat([b.speculative_ids for b in batches], dim=0)
            if batches[0].speculative_ids is not None
            else None
        )

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            start_slots=start_slots,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            slots=slots,
            max_seqlen=max_seqlen,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            speculative_ids=speculative_ids,
        )

    def __len__(self):
        return len(self.requests)


class FlashCausalLM(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        sliding_window: Optional[int] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

        self.cuda_graphs = {}
        self.kv_cache = []

        super(FlashCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=sliding_window,
        )

    @property
    def batch_type(self) -> Type[FlashCausalLMBatch]:
        return FlashCausalLMBatch

    def max_past(self) -> int:
        return getattr(self.model, "max_past", None)

    def init_kv_cache(
        self,
        num_blocks: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.kv_cache = []
        empty_cache()

        element_size = torch.tensor([], dtype=dtype).element_size()
        if SYSTEM == "xpu":
            x = 1
        else:
            x = BLOCK_SIZE // element_size

        self.kv_cache = [
            (
                torch.empty(
                    (num_blocks, num_heads, head_size // x, BLOCK_SIZE, x),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_blocks, num_heads, head_size, BLOCK_SIZE),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(num_layers)
        ]

    def cuda_graph_warmup(self, bs: int, max_s: int, max_bt: int):
        input_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
        position_ids = torch.zeros(bs, dtype=torch.int32, device=self.device)
        slots = torch.arange(bs, dtype=torch.int64, device=self.device)
        input_lengths = torch.ones(bs, dtype=torch.int32, device=self.device) * max_s
        block_tables = (
            torch.arange(max_bt, dtype=torch.int32, device=self.device)
            .repeat(bs)
            .reshape((bs, max_bt))
        )

        self.cuda_graphs[bs] = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "kv_cache": self.kv_cache,
            "block_tables": block_tables,
            "slots": slots,
            "input_lengths": input_lengths,
        }
        graph = torch.cuda.CUDAGraph()
        self.cuda_graphs[bs]["graph"] = graph

        torch.cuda.synchronize()
        # Run once outside to warmup
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            kv_cache=self.kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            prefill_cache_indices=None,
            lm_head_indices=None,
        )
        torch.cuda.synchronize()

        with torch.cuda.graph(graph, pool=MEM_POOL):
            logits, speculative_logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=None,
                kv_cache=self.kv_cache,
                block_tables=block_tables,
                slots=slots,
                input_lengths=input_lengths,
                max_s=max_s,
                prefill_cache_indices=None,
                lm_head_indices=None,
            )
            self.cuda_graphs[bs]["logits"] = logits
            self.cuda_graphs[bs]["speculative_logits"] = speculative_logits
        torch.cuda.synchronize()

    def warmup(self, batch: FlashCausalLMBatch):
        # The warmup batch is the biggest batch we could ever receive
        empty_cache()

        try:
            self.init_kv_cache(
                batch.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.dtype,
                self.device,
            )
            max_bt = batch.max_blocks
            max_s = max_bt * BLOCK_SIZE

            if SYSTEM == "rocm" and os.environ.get("PYTORCH_TUNABLEOP_ENABLED", False):
                torch.cuda.tunable.tuning_enable(False)
            _, batch, _ = self.generate_token(batch)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            ) from e

        synchronize(self.device)

        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
        total_cache_size = self.num_layers * cache_block_size * 2 * dtype_size

        free_memory = get_free_memory(self.device, MEMORY_FRACTION)

        num_blocks = (
            # Leave 5% for some wiggle room
            int((free_memory * 0.95) // total_cache_size)
            # Add batch.num_blocks as we allocated it above, so it is included in the peak memory.
            + batch.num_blocks
        )

        del batch

        self.init_kv_cache(
            num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.dtype,
            self.device,
        )

        if SYSTEM == "rocm":
            if (
                os.environ.get("PYTORCH_TUNABLEOP_ENABLED") is None
                or os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1"
            ):
                if os.environ.get("PYTORCH_TUNABLEOP_TUNING") != "0":
                    torch.cuda.tunable.tuning_enable(True)

                if os.environ.get("PYTORCH_TUNABLEOP_SEQLENS") is not None:
                    tuning_sequences = [
                        int(val)
                        for val in os.environ["PYTORCH_TUNABLEOP_SEQLENS"].split(",")
                    ]
                else:
                    tuning_sequences = CUDA_GRAPHS

                tunableop_filepath = os.path.join(
                    HUGGINGFACE_HUB_CACHE,
                    f"tunableop_{tgi_globals.MODEL_ID.replace('/', '-')}_tp{self.world_size}_rank{self.rank}.csv",
                )

                logger.info(
                    f"PyTorch TunableOp (https://github.com/fxmarty/pytorch/tree/2.3-patched/aten/src/ATen/cuda/tunable) is enabled. The warmup may take several minutes, picking the ROCm optimal matrix multiplication kernel for the target lengths {', '.join([str(seqlen) for seqlen in tuning_sequences])}, with typical 5-8% latency improvement for small sequence lengths. The picked GEMMs are saved in the file {tunableop_filepath}. To disable TunableOp, please launch TGI with `PYTORCH_TUNABLEOP_ENABLED=0`."
                )

                if os.path.isfile(tunableop_filepath):
                    logger.info(
                        f"The file {tunableop_filepath} already exists and will be reused."
                    )
                    torch.cuda.tunable.read_file(tunableop_filepath)

                os.makedirs(HUGGINGFACE_HUB_CACHE, exist_ok=True)

                for seqlen in tuning_sequences:
                    logger.info(f"Warming up TunableOp for seqlen={seqlen}")
                    self.tunableop_warmup(seqlen)
                    torch.cuda.tunable.write_file(tunableop_filepath)
                torch.cuda.tunable.tuning_enable(False)
            else:
                logger.info(
                    "PyTorch ROCm TunableOp (https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable) is disabled. TunableOp brings an additional 5-8% latency improvement for small sequence lengths but requires a warmup. If necessary, please use the environment variable PYTORCH_TUNABLEOP_ENABLED=1 to enable TunableOp."
                )

        if CUDA_GRAPHS:
            try:
                logger.info(f"Cuda Graphs are enabled for sizes {CUDA_GRAPHS}")
                # Warmup cuda graphs
                for bs in CUDA_GRAPHS:
                    if self.speculate is None or self.speculate + 1 <= bs:
                        self.cuda_graph_warmup(bs, max_s, max_bt)
            except torch.cuda.OutOfMemoryError:
                logger.exception(f"Decode cuda graph warmup failed")
        else:
            logger.info(f"Cuda Graphs are disabled (CUDA_GRAPHS={CUDA_GRAPHS}).")

        return int(num_blocks * BLOCK_SIZE)

    def tunableop_warmup(self, seqlen: int):
        input_ids = torch.zeros(seqlen, dtype=torch.int64, device=self.device)
        position_ids = torch.zeros(seqlen, dtype=torch.int32, device=self.device)
        slots = torch.arange(seqlen, dtype=torch.int64, device=self.device)

        # Dummy value, some models (starcoder2) don't accept `None`.
        input_lengths = torch.ones(seqlen, dtype=torch.int32, device=self.device)

        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=torch.tensor(
                [0, seqlen], device=self.device, dtype=torch.int32
            ),
            kv_cache=self.kv_cache,
            block_tables=None,
            input_lengths=input_lengths,
            slots=slots,
            max_s=seqlen,
            lm_head_indices=None,
            prefill_cache_indices=None,
        )

    def forward(
        self, batch: FlashCausalLMBatch
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
            max_s = batch.max_seqlen
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
            max_s = batch.max_seqlen
            lm_head_indices = batch.prefill_head_indices

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        bs = input_ids.shape[0]
        sorted_padded_bs = sorted([k for k in self.cuda_graphs.keys() if k >= bs])
        if sorted_padded_bs:
            # Get associated cuda graph
            cuda_graph = self.cuda_graphs[sorted_padded_bs[0]]
        else:
            cuda_graph = None

        if cu_seqlen_prefill is not None or cuda_graph is None:
            logits, speculative_logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=cu_seqlen_prefill,
                kv_cache=kv_cache,
                block_tables=block_tables,
                slots=slots,
                input_lengths=input_lengths,
                max_s=max_s,
                prefill_cache_indices=batch.prefill_cache_indices,
                lm_head_indices=lm_head_indices,
            )
            if batch.prefill_cache_indices is not None:
                batch.prefill_cache_indices = None
            return logits, speculative_logits

        # Copy inputs to the static inputs of the cuda graph
        # Static inputs are potentially padded
        cuda_graph["input_ids"][: input_ids.shape[0]] = input_ids
        cuda_graph["position_ids"][: position_ids.shape[0]] = position_ids
        cuda_graph["block_tables"][
            : block_tables.shape[0], : block_tables.shape[1]
        ] = block_tables
        cuda_graph["slots"].fill_(-1)
        cuda_graph["slots"][: slots.shape[0]] = slots
        cuda_graph["input_lengths"].zero_()
        cuda_graph["input_lengths"][: input_lengths.shape[0]] = input_lengths

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

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch], Tuple[int, int]]:
        start = time.time_ns()
        prefill = batch.cu_seqlen_prefill is not None
        prefill_logprobs = batch.prefill_next_token_indices is not None

        out, speculative_logits = self.forward(batch)

        if prefill:
            next_token_logits = (
                out[batch.prefill_next_token_indices] if prefill_logprobs else out
            )
            if speculative_logits is not None:
                speculative_logits = (
                    speculative_logits[batch.prefill_next_token_indices]
                    if prefill_logprobs
                    else speculative_logits
                )
        else:
            next_token_logits = out

        speculate = get_speculate()
        (
            next_input_ids,
            next_token_logprobs,
            logprobs,
            accepted_ids,
            speculative_ids,
        ) = batch.next_token_chooser(
            batch.all_input_ids_tensor[:, : batch.max_seqlen],
            next_token_logits,
            speculate,
            batch.speculative_ids,
            speculative_logits,
        )

        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens, batch.top_n_tokens_tensor, logprobs, accepted_ids
        )

        if prefill:
            if len(batch) > 1 and prefill_logprobs:
                # We create the prefill_tokens_indices tensor that will be used to gather prefill logprobs
                # When batch == 1, we will just use the batch.input_ids values directly
                prefill_tokens_indices = batch.input_ids.new_zeros(len(out))

            next_position_ids = batch.position_ids.new_empty(len(batch))
            batch.slot_indices = batch.slot_indices[batch.cu_seqlen_prefill[1:] - 1]
            # We do not need cu_seqlen_prefill anymore
            batch.cu_seqlen_prefill = None
        else:
            prefill_logprobs = None
            next_position_ids = batch.position_ids

        # Cumulative length
        cumulative_length = 0

        # Results
        generations: List[Generation] = []
        stopped = True

        # Zipped iterator
        iterator = zip(batch.input_lengths, batch.all_input_ids, accepted_ids)

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        index = 0
        for i, (input_length, all_input_ids, n_accepted_ids) in enumerate(iterator):
            # Indexing metadata
            start_index = cumulative_length
            end_index = cumulative_length + input_length

            if prefill:
                # Indexing metadata
                out_start_index = batch.prefill_cu_outlens[i]
                out_end_index = batch.prefill_cu_outlens[i + 1]
                out_length = out_end_index - out_start_index

                # Initialize position_ids
                # In decode, we do not need this as we can just increment position ids
                next_position_ids[i] = batch.position_ids[end_index - 1]

                # Used to gather prefill logprobs
                # Copy batch.input_ids to prefill_token_indices
                if prefill_logprobs:
                    if len(batch) > 1:
                        prefill_tokens_indices[out_start_index : out_end_index - 1] = (
                            batch.input_ids[start_index + 1 : start_index + out_length]
                        )
                    else:
                        # Set prefill_tokens_indices to the correct slice
                        prefill_tokens_indices = batch.input_ids[
                            start_index + 1 : start_index + out_length
                        ]

            for j in range(n_accepted_ids):
                batch.all_input_ids_tensor[i, input_length + j] = next_input_ids[index]
                index += 1

            cumulative_length += input_length

        # Update values
        batch.input_ids = next_input_ids[accepted_ids.cumsum(dim=-1) - 1]
        batch.speculative_ids = speculative_ids
        batch.position_ids = next_position_ids + accepted_ids
        batch.input_lengths_tensor += accepted_ids
        batch.slot_indices += accepted_ids

        if prefill and prefill_logprobs:
            # Get prefill logprobs
            prefill_logprobs_tensor = torch.log_softmax(out, -1)
            prefill_logprobs = torch.gather(
                prefill_logprobs_tensor, 1, prefill_tokens_indices.view(-1, 1)
            )
            # GPU <-> CPU sync
            prefill_logprobs = prefill_logprobs.view(-1).tolist()

        # GPU <-> CPU sync
        next_token_logprobs = next_token_logprobs.tolist()
        next_token_ids = next_input_ids.tolist()
        accepted_ids = accepted_ids.tolist()
        start_decode = time.time_ns()

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.next_token_chooser.do_sample,
            batch.next_token_chooser.seeds,
            batch.top_n_tokens,
            accepted_ids,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        # For each member of the batch
        index = 0
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            stopping_criteria,
            all_input_ids,
            do_sample,
            seed,
            top_n_tokens,
            n_accepted_ids,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Append next token to all tokens
            next_token_texts = []
            left = 0

            if n_accepted_ids > 1:
                if RANK == 0:
                    logger.debug(f"Speculated ids {n_accepted_ids - 1}")

            current_stopped = False
            for j in range(index, index + n_accepted_ids):
                # Generated token
                next_token_id = next_token_ids[j]
                all_input_ids.append(next_token_id)
                next_token_text, prefix_offset, read_offset = self.decode_token(
                    all_input_ids,
                    prefix_offset,
                    read_offset,
                )
                next_token_texts.append(next_token_text)

                stop, reason = stopping_criteria(
                    next_token_id,
                    next_token_text,
                )

                if stop:
                    left = index + n_accepted_ids - j - 1
                    current_stopped = True
                    break
                else:
                    current_stopped = False
            stopped = stopped and current_stopped

            _next_token_ids = next_token_ids[index : index + n_accepted_ids - left]
            _next_token_logprobs = next_token_logprobs[
                index : index + n_accepted_ids - left
            ]
            index += n_accepted_ids

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text, _, _ = self.decode_token(
                        all_input_ids,
                        prefix_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens
                        - 1,
                        read_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens,
                        skip_special_tokens=True,
                    )
                    generated_text = GeneratedText(
                        output_text,
                        stopping_criteria.current_tokens,
                        reason,
                        seed if do_sample else None,
                    )
                else:
                    generated_text = None

                # Prefill
                if prefill and request.prefill_logprobs:
                    out_start_index = batch.prefill_cu_outlens[i]
                    out_end_index = batch.prefill_cu_outlens[i + 1]

                    # Remove generated token to only have prefill and add nan for first prompt token
                    request_prefill_logprobs = [float("nan")] + prefill_logprobs[
                        out_start_index : out_end_index - 1
                    ]
                    prefill_token_ids = all_input_ids[:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )

                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        request_prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        _next_token_ids,
                        _next_token_logprobs,
                        next_token_texts,
                        [nid in self.all_special_ids for nid in _next_token_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            # accept each new token for this specific request since we may
            # have more than one new token per request with speculative decoding
            for next_token_id in _next_token_ids:
                batch.next_token_chooser = (
                    batch.next_token_chooser.advance_grammar_single(i, next_token_id)
                )

            # Update values
            batch.input_lengths[i] = input_length + n_accepted_ids
            if batch.input_lengths[i] > batch.max_seqlen:
                batch.max_seqlen = batch.input_lengths[i]
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids

        if stopped:
            # No need to return a batch if we know that all requests stopped
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        batch.prefill_cu_outlens = None
        batch.prefill_head_indices = None
        batch.prefill_next_token_indices = None

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)
