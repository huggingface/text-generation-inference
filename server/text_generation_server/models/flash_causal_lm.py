import math
import itertools
import torch
import torch.distributed

import numpy as np

from dataclasses import dataclass
from opentelemetry import trace
from transformers import PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Union, Dict

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import StoppingCriteria, HeterogeneousNextTokenChooser
from text_generation_server.utils.dist import MEMORY_FRACTION

tracer = trace.get_tracer(__name__)

BLOCK_SIZE = 16
# Will be set in warmup
CACHE_MANAGER: Optional["CacheManager"] = None


class CacheManager:
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.block_size = BLOCK_SIZE
        self.num_blocks = num_blocks

        element_size = torch.tensor([], dtype=dtype).element_size()
        x = self.block_size // element_size

        self.kv_cache = [
            (
                torch.empty(
                    (num_blocks, num_heads, head_size // x, self.block_size, x),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_blocks, num_heads, head_size, self.block_size),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(num_layers)
        ]
        self.free_block_mask = torch.ones(num_blocks, dtype=torch.int32, device="cpu")
        self.slots = torch.arange(
            0, num_blocks * self.block_size, dtype=torch.int32
        ).view(num_blocks, self.block_size)

    def allocate(self, batch: "FlashCausalLMBatch"):
        # Get free blocks indices by finding values in mask that are not set to 0
        free_block_indices = self.free_block_mask.nonzero()
        assert (
            len(free_block_indices) >= batch.blocks
        ), f"Out of available cache blocks: asked {batch.blocks}, only {len(free_block_indices)} free blocks"

        # Slice by the number of required blocks
        block_indices = free_block_indices[: batch.blocks]
        block_indices = block_indices.flatten()

        # Padded block tables
        block_tables_tensor = torch.zeros(
            (len(batch), batch.max_blocks), dtype=torch.int32
        )

        # Allocate paged attention blocks
        cumulative_blocks = 0
        slots = []
        block_tables = []
        for i, (needed_blocks, needed_slots) in enumerate(batch.needed_blocks_slots):
            # Get allocated blocks for this sequence
            allocated_blocks = block_indices[
                cumulative_blocks : cumulative_blocks + needed_blocks
            ]
            # Get slots for the allocated blocks
            allocated_slots = self.slots[allocated_blocks].flatten()[:needed_slots]

            slots.append(allocated_slots)
            block_tables.append(allocated_blocks.tolist())
            block_tables_tensor[i, :needed_blocks] = allocated_blocks
            cumulative_blocks += needed_blocks

        batch.needed_blocks_slots = None
        batch.block_tables = block_tables
        batch.block_tables_tensor = block_tables_tensor.to(batch.input_ids.device)
        batch.slots = torch.concat(slots).to(batch.input_ids.device)

        # Allocate the required number of blocks by setting the mask to 0
        self.free_block_mask[block_indices] = 0

    def free(self, block_indices: Optional[List[int]]):
        if block_indices is not None and block_indices:
            # Reset mask
            self.free_block_mask[block_indices] = 1


@dataclass
class FlashCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    # request id -> idx in list mapping
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor
    position_ids: torch.Tensor

    # Flash Attention values

    # tensor of length b containing the cumulative sequence lengths of the sequences in the batch, only used in prefill
    cu_seqlen_prefill: Optional[torch.Tensor]

    # Paged Attention values

    # Set when creating the batch
    # CPU tensor of length b indicating the start of each sequence in slots
    start_slots: torch.Tensor
    # tensor of indices of the currently used slots, length = \sum_{i=0}^{b} s_i in prefill, length = b in decode
    slot_indices: torch.Tensor
    # List of tuple of ints representing the number of blocks and slots needed by each sequence
    needed_blocks_slots: Optional[List[Tuple[int, int]]]

    # Set in prefill by the CacheManager
    # list of length b of list of length s_i // block_size
    block_tables: Optional[List[List[int]]]
    # tensor of size [b, max_seqlen // block_size] holding the paged attention block tables for all sequences
    block_tables_tensor: Optional[torch.Tensor]
    # tensor of length \sum_{i=0}^{b} max_s_i  holding the paged attention slots for all sequences
    slots: Optional[torch.Tensor]

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

    # Number of blocks in this batch
    blocks: int
    # Maximum number of blocks
    max_blocks: int

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.blocks * BLOCK_SIZE,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        batch_inputs = []
        max_truncation = 0
        for r in pb.requests:
            batch_inputs.append(r.inputs)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_truncation
        )["input_ids"]

        position_ids = []
        cu_seqlen_prefill = [0]
        needed_blocks_slots = []
        start_slots = []
        slot_indices = []

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

        # Cumulative length
        cumulative_length = 0
        cumulative_max_length = 0
        prefill_out_cumulative_length = 0

        blocks = 0
        max_seqlen = 0
        max_length = 0
        max_blocks = 0

        # Parse batch
        for i, (r, tokenized_input) in enumerate(
            zip(pb.requests, batch_tokenized_inputs)
        ):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenized_input[-r.truncate :]

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

            # Paged attention
            # Remove one as the first token des not have a past
            total_tokens = input_length + max_new_tokens - 1
            needed_blocks = math.ceil(total_tokens / BLOCK_SIZE)
            blocks += needed_blocks
            needed_blocks_slots.append((needed_blocks, total_tokens))
            start_slots.append(cumulative_max_length)

            request_slot_indices = torch.arange(
                cumulative_max_length,
                cumulative_max_length + input_length,
                dtype=torch.int64,
            )
            slot_indices.append(request_slot_indices)

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
            max_blocks = max(max_blocks, needed_blocks)
            max_length = max(max_length, input_length + max_new_tokens)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype, device
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
        else:
            input_ids = all_input_ids[0]
            position_ids = position_ids[0]
            slot_indices = slot_indices[0]

        cu_seqlen_prefill = torch.tensor(
            cu_seqlen_prefill, device=device, dtype=torch.int32
        )

        position_ids = position_ids.to(device)
        slot_indices = slot_indices.to(device)
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

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            start_slots=start_slots,
            slot_indices=slot_indices,
            needed_blocks_slots=needed_blocks_slots,
            block_tables=None,
            block_tables_tensor=None,
            slots=None,
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
            blocks=blocks,
            max_blocks=max_blocks,
        )

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

        blocks = 0
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

            remaining_tokens = (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )

            request_block_table = self.block_tables[idx]
            blocks += len(request_block_table)
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

        global CACHE_MANAGER
        block_indices_to_free = []
        # Iterate on all requests
        for i, r in enumerate(self.requests):
            # Filter requests that are not part of the new batch
            if r.id not in requests_idx_mapping.keys():
                block_indices_to_free.extend(self.block_tables[i])
        # Free blocks
        CACHE_MANAGER.free(block_indices_to_free)
        # Needed to avoid dropping blocks when the batches will go out of scope
        self.block_tables = None

        # Index into tensors
        input_ids = self.input_ids[indices]
        position_ids = self.position_ids[indices]
        all_input_ids_tensor = self.all_input_ids_tensor[indices]
        block_tables_tensor = self.block_tables_tensor[indices]
        input_lengths_tensor = self.input_lengths_tensor[indices]
        slots = self.slots[slot_filtering_indices]
        next_token_chooser = self.next_token_chooser.filter(indices)

        start_slots = torch.tensor(start_slots, dtype=torch.int64)

        # Move to GPU now that we have the whole tensor
        slot_indices = slot_indices.to(device)

        return FlashCausalLMBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            start_slots=start_slots,
            slot_indices=slot_indices,
            needed_blocks_slots=None,
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
            blocks=blocks,
            max_blocks=max_blocks,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        requests_idx_mapping = {}

        blocks = 0
        total_batch_size = 0
        total_slots = 0
        max_blocks = 0
        max_length = 0
        max_seqlen = 0
        for b in batches:
            total_batch_size += len(b)
            total_slots += len(b.slots)
            blocks += b.blocks
            max_blocks = max(max_blocks, b.max_blocks)
            max_seqlen = max(max_seqlen, b.max_seqlen)
            max_length = max(
                max_length,
                max(
                    input_length
                    + stopping_criteria.max_new_tokens
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

        start_slots = []
        block_tables = []
        all_input_ids = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        next_token_chooser_parameters = []
        stopping_criterias = []

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
            stopping_criterias.extend(batch.stopping_criterias)

            # Update
            cumulative_batch_size += len(batch)
            cumulative_slots += len(batch.slots)

        start_slots = torch.concat(start_slots)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters,
            dtype=batches[0].next_token_chooser.dtype,
            device=batches[0].next_token_chooser.device,
        )

        # Needed to avoid dropping blocks when the batches will go out of scope
        for b in batches:
            b.block_tables = None
            del b

        return FlashCausalLMBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            start_slots=start_slots,
            slot_indices=slot_indices,
            needed_blocks_slots=None,
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
            blocks=blocks,
            max_blocks=max_blocks,
        )

    def __del__(self):
        if self.block_tables is not None and self.block_tables:
            global CACHE_MANAGER
            # Free blocks
            CACHE_MANAGER.free(list(itertools.chain.from_iterable(self.block_tables)))

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
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

        super(FlashCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    @property
    def batch_type(self) -> Type[FlashCausalLMBatch]:
        return FlashCausalLMBatch

    def warmup(self, batch: FlashCausalLMBatch):
        global CACHE_MANAGER

        torch.cuda.empty_cache()
        try:
            CACHE_MANAGER = CacheManager(
                batch.blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.dtype,
                self.device,
            )
            _, batch = self.generate_token(batch)
        except Exception as e:
            raise RuntimeError(
                f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            ) from e

        torch.cuda.synchronize(self.device)

        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
        total_cache_size = self.num_layers * cache_block_size * 2 * dtype_size

        total_free_memory, _ = torch.cuda.mem_get_info(self.device)
        total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory

        free_memory = max(
            0, total_free_memory - (1 - MEMORY_FRACTION) * total_gpu_memory
        )

        num_blocks = (
            int(free_memory // total_cache_size)
            # Add batch.blocks as we allocated it above, so it is included in the peak memory.
            + CACHE_MANAGER.num_blocks
        )

        del CACHE_MANAGER
        del batch
        torch.cuda.empty_cache()

        CACHE_MANAGER = CacheManager(
            num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.dtype,
            self.device,
        )

        return int(num_blocks * BLOCK_SIZE)

    def decode(self, generated_ids: Union[torch.Tensor, List[int]]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        global CACHE_MANAGER

        # Model Forward
        return self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=CACHE_MANAGER.kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            lm_head_indices=lm_head_indices,
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch]]:
        prefill = batch.cu_seqlen_prefill is not None
        prefill_logprobs = batch.prefill_next_token_indices is not None

        if batch.needed_blocks_slots:
            # Allocate blocks to this batch
            CACHE_MANAGER.allocate(batch)

        try:
            out = self.forward(
                batch.input_ids,
                batch.position_ids,
                batch.cu_seqlen_prefill,
                batch.block_tables_tensor,
                batch.slots[batch.slot_indices],
                batch.input_lengths_tensor,
                batch.max_seqlen,
                batch.prefill_head_indices,
            )
        except Exception as e:
            del batch
            raise e

        if prefill:
            next_token_logits = (
                out[batch.prefill_next_token_indices] if prefill_logprobs else out
            )
        else:
            next_token_logits = out

        next_input_ids, next_token_logprobs = batch.next_token_chooser(
            batch.all_input_ids_tensor[:, : batch.max_seqlen], next_token_logits
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
        iterator = zip(
            batch.input_lengths,
            batch.all_input_ids,
        )

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        for i, (
            input_length,
            all_input_ids,
        ) in enumerate(iterator):
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
                        prefill_tokens_indices[
                            out_start_index : out_end_index - 1
                        ] = batch.input_ids[start_index + 1 : start_index + out_length]
                    else:
                        # Set prefill_tokens_indices to the correct slice
                        prefill_tokens_indices = batch.input_ids[
                            start_index + 1 : start_index + out_length
                        ]

            batch.all_input_ids_tensor[i, input_length] = next_input_ids[i]

            cumulative_length += input_length

        # Set values in batch
        batch.input_ids = next_input_ids
        batch.position_ids = next_position_ids + 1
        batch.input_lengths_tensor += 1
        batch.slot_indices += 1

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
        next_token_ids = batch.input_ids.tolist()

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
            next_token_ids,
            next_token_logprobs,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            stopping_criteria,
            all_input_ids,
            do_sample,
            seed,
            next_token_id,
            next_token_logprob,
        ) in enumerate(iterator):
            # Append next token to all tokens
            all_input_ids.append(next_token_id)

            # Generated token
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids,
                prefix_offset,
                read_offset,
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text = self.decode(
                        all_input_ids[-stopping_criteria.current_tokens :]
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
                    prefill_tokens = PrefillTokens(
                        prefill_token_ids, request_prefill_logprobs, prefill_texts
                    )
                else:
                    prefill_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    next_token_id,
                    next_token_logprob,
                    next_token_text,
                    next_token_id in self.all_special_ids,
                    generated_text,
                )

                generations.append(generation)

            # Update values
            batch.input_lengths[i] = input_length + 1
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids

        if stopped:
            del batch
            # No need to return a batch if we know that all requests stopped
            return generations, None

        batch.prefill_cu_outlens = None
        batch.prefill_head_indices = None
        batch.prefill_next_token_indices = None
        batch.max_seqlen = batch.max_seqlen + 1

        return generations, batch
