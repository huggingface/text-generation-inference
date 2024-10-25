from contextlib import nullcontext
import math
import os
import time
import torch
import torch.distributed

import numpy as np

from loguru import logger
from dataclasses import dataclass
from opentelemetry import trace
from transformers import (
    PreTrainedTokenizerBase,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
)
from typing import (
    Any,
    ContextManager,
    Iterable,
    Optional,
    Tuple,
    List,
    Type,
    Dict,
    Union,
)

from text_generation_server.adapters import AdapterBatchData, AdapterBatchMetadata
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from text_generation_server.utils.chunks import concat_text_chunks
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models import Model
from text_generation_server.utils.log import log_master
from text_generation_server.utils.prefill_chunking import (
    get_support_chunking,
    get_max_prefill_tokens,
)
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.models.globals import (
    MEM_POOL,
    ATTENTION,
    BLOCK_SIZE,
    CUDA_GRAPHS,
    TGI_WIGGLE_ROOM,
    get_adapter_to_index,
)
from text_generation_server.layers.attention import KVCache, Seqlen
from text_generation_server.utils import StoppingCriteria, HeterogeneousNextTokenChooser
from text_generation_server.utils.dist import MEMORY_FRACTION
from text_generation_server.utils.quantization import get_loader
from text_generation_server.utils.segments import SegmentConcatBuilder, find_segments

from text_generation_server.utils.import_utils import (
    empty_cache,
    synchronize,
    get_free_memory,
)
from text_generation_server.models.metadata_kernels import (
    has_triton,
    copy_next_input_ids_inplace,
    block_tables_to_ragged,
    block_tables_to_padded,
    prepare_position_slot_ids,
    slots_filtering,
)

tracer = trace.get_tracer(__name__)

# Will be set in init
SLIDING_WINDOW: Optional[int] = None


def set_sliding_window(sliding_window: int):
    global SLIDING_WINDOW
    SLIDING_WINDOW = sliding_window


def get_sliding_windows() -> int:
    global SLIDING_WINDOW
    return SLIDING_WINDOW


def init_cpu_threads_env(rank_id: int, world_size: int):
    import importlib.util

    if importlib.util.find_spec("numa") is not None:
        import numa
        import psutil

        nodes = numa.info.get_max_node() + 1
        rank_per_node = math.ceil(world_size / nodes)
        num_cpus_per_nodes = int(psutil.cpu_count(logical=False) / nodes)
        node_id = int(rank_id / rank_per_node)
        rank_offset_per_node = rank_id % rank_per_node
        if os.getenv("OMP_NUM_THREADS") is None:
            num_cpus_per_rank = max(int(num_cpus_per_nodes / rank_per_node), 1)
        else:
            num_cpus_per_rank = int(os.getenv("OMP_NUM_THREADS"))
        if len(numa.memory.get_membind_nodes()) == nodes:
            numa.memory.set_membind_nodes((node_id))
        torch.set_num_threads(num_cpus_per_rank)
        if len(numa.schedule.get_affinitive_cpus(0)) == psutil.cpu_count(logical=True):
            cpu_start = num_cpus_per_rank * rank_offset_per_node
            numa.schedule.run_on_cpus(
                0,
                *(
                    numa.info.node_to_cpus(node_id)[
                        cpu_start : cpu_start + num_cpus_per_rank
                    ]
                ),
            )
        logger.info(
            f"affinity={numa.schedule.get_affinitive_cpus(0)}, membind = {numa.memory.get_membind_nodes()}"
        )


@dataclass
class FlashCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    # request id -> idx in list mapping
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    # Can be a list for easy filtering
    # If `input_ids` is a list, it needs to be materialized to a tensor first
    input_ids: Union[torch.Tensor, List[List[int]]]
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    position_ids: Optional[torch.Tensor]
    speculative_ids: Optional[torch.Tensor]

    # Set when creating the batch
    # tensor of indices of the currently used slots, length = \sum_{i=0}^{b} s_i in prefill, length = b in decode
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    slot_indices: Optional[torch.Tensor]

    # list of length b of list of length s_i // block_size
    block_tables: List[List[int]]
    # tensor of size [b, max_total_seqlen // block_size] holding the paged attention block tables for all sequences
    block_tables_tensor: torch.Tensor
    # tensor of length \sum_{i=0}^{b} max_s_i  holding the paged attention slots for all sequences
    slots: torch.Tensor
    # list of length b + 1  containing the cumulative sequence slot lengths of the sequences in the batch
    # used for filtering
    cu_slots: torch.Tensor

    max_input_length: int
    max_current_length: int

    # Whether this batch contains at least one request that is prefilling
    prefilling: bool
    # Whether each request is prefilling
    prefilling_mask: List[bool]

    # Prefill metadata tensors to efficiently compute logprobs
    # tensor of length b + 1  containing the cumulative sequence lengths of the sequences in the batch, only used in prefill
    cu_seqlen_prefill: Optional[torch.Tensor]
    # Prefill cache indices is used to slice into the kv tensor before caching it into the paged attention buffers
    # as we only keep SLIDING_WINDOW values instead of the whole tensor
    prefill_cache_indices: Optional[torch.Tensor]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_head_indices: Optional[torch.Tensor]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_next_token_indices: Optional[torch.tensor]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_cu_outlens: Optional[List[int]]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_logprob_tokens: List[Optional[Tokens]]

    # All tokens
    all_input_ids: List[List[int]]
    all_input_ids_tensor: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    # size [b], containing the number of blocks that can be retrieved from the cache
    cache_lengths: List[int]
    prompt_lengths: List[int]
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    input_lengths_tensor: Optional[torch.Tensor]
    cache_lengths_tensor: Optional[torch.Tensor]
    prompt_lengths_tensor: torch.Tensor

    prefix_offsets: List[Optional[int]]
    read_offsets: List[Optional[int]]

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    stopping_criterias: List[StoppingCriteria]
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    # Adapter metadata for each request
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    adapter_meta: Optional[AdapterBatchMetadata]

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
            current_tokens=(
                sum([len(i) for i in self.input_ids])
                if isinstance(self.input_ids, list)
                else len(self.input_ids)
            ),
        )

    @classmethod
    def batch_tokenized_inputs(
        cls, requests: Iterable[generate_pb2.Request], tokenizer
    ):
        max_length = 0
        all_input_ids = []
        batch_size = 0
        for r in requests:
            batch_size += 1
            inputs = concat_text_chunks(r.input_chunks.chunks)
            input_ids = tokenizer(
                inputs,
                truncation=True,
                max_length=r.truncate,
                add_special_tokens=r.add_special_tokens,
            )["input_ids"]
            max_length = max(max_length, len(input_ids))
            all_input_ids.append(input_ids)
        return all_input_ids

    @classmethod
    def from_tokenized(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        batch_tokenized_inputs,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        speculate = get_speculate()

        cache_lengths = []
        input_lengths = []
        prompt_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        all_postfix_ids = []
        requests_idx_mapping = {}
        slots = []
        cu_slots = [0]

        next_token_chooser_parameters = []
        stopping_criterias = []
        top_n_tokens = []

        num_blocks = 0
        max_input_length = 0
        max_current_length = 0
        max_length = 0
        max_blocks = 0

        cu_blocks = [0]
        block_tables = []
        block_tables_ragged = []

        # Parse batch
        for i, (r, tokenized_input) in enumerate(
            zip(pb.requests, batch_tokenized_inputs)
        ):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            prompt_length = len(tokenized_input)
            prompt_lengths.append(prompt_length)

            cache_length = r.cache_len

            assert (
                cache_length <= prompt_length
            ), f"Prefix {cache_length} vs input {prompt_length}"
            if cache_length == prompt_length:
                assert False, "unreachable"

            # `chunk_len` is an optional field in the protobuf
            # It is only set if the model support chunking
            if r.HasField("chunk_len"):
                input_length = r.chunk_len

                if cache_length + input_length < prompt_length:
                    # FIXME: speculate is not supported for context chunking at the moment
                    assert speculate == 0
                    assert get_support_chunking()
                    assert input_length > 0

                postfix_ids = tokenized_input[
                    cache_length : cache_length + input_length
                ]
                assert (
                    len(postfix_ids) == input_length
                ), "Rust and Python tokenizers are not aligned"
            else:
                # Use all the remaining ids
                postfix_ids = tokenized_input[cache_length:]
                input_length = len(postfix_ids)

            input_lengths.append(input_length)

            prefix_offsets.append(prompt_length - 5)
            read_offsets.append(prompt_length)

            all_postfix_ids.append(postfix_ids)
            all_input_ids.append(tokenized_input)

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

            # Tokens that need to be mapped to blocks.
            block_tokens = prompt_length + max_new_tokens - 1 + speculative_length

            # blocks and slots can be empty (for example in warmup)
            if not r.blocks:
                needed_blocks = math.ceil(block_tokens / BLOCK_SIZE)
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
            block_tables_ragged.extend(request_blocks)
            cu_blocks.append(len(block_tables_ragged))

            slots.extend(request_slots)
            cu_slots.append(len(slots))

            cache_lengths.append(cache_length)
            num_blocks += len(request_blocks)

            # Update
            max_blocks = max(max_blocks, len(request_blocks))
            max_input_length = max(max_input_length, input_length)
            max_current_length = max(max_current_length, cache_length + input_length)
            max_length = max(
                max_length,
                prompt_length + max_new_tokens + speculative_length,
            )

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype, device, tokenizer
        )

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

        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        block_tables_ragged = torch.tensor(
            block_tables_ragged, device=device, dtype=torch.int32
        )
        cu_blocks = torch.tensor(cu_blocks, device=device, dtype=torch.int64)
        block_tables_tensor = torch.empty(
            (len(block_tables), max_blocks),
            device=device,
            dtype=torch.int32,
        )

        # If the device supports Triton, we can use a fused kernel
        if has_triton():
            block_tables_to_padded(
                max_blocks, cu_blocks, block_tables_tensor, block_tables_ragged
            )
        else:
            for i, request_blocks in enumerate(block_tables):
                block_tables_tensor[i, : len(request_blocks)] = torch.tensor(
                    request_blocks
                )

        prompt_lengths_tensor = torch.tensor(
            prompt_lengths, dtype=torch.int32, device=device
        )

        slots = torch.tensor(slots, dtype=torch.int64, device=device)
        cu_slots = torch.tensor(cu_slots, dtype=torch.int64)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=all_postfix_ids,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            cache_lengths=cache_lengths,
            max_input_length=max_input_length,
            max_current_length=max_current_length,
            prefilling=True,
            prefilling_mask=[True] * len(pb.requests),
            prefill_logprob_tokens=[None] * len(pb.requests),
            input_lengths=input_lengths,
            prompt_lengths=prompt_lengths,
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
            prompt_lengths_tensor=prompt_lengths_tensor,
            # These values will be set by `FlashCausalLMBatch.prepare_for_prefill`
            position_ids=None,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            slot_indices=None,
            slots=slots,
            cu_slots=cu_slots,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            cache_lengths_tensor=None,
            input_lengths_tensor=None,
            adapter_meta=None,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        assert len(pb.requests) > 0
        batch_tokenized_inputs = cls.batch_tokenized_inputs(pb.requests, tokenizer)
        return cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> "FlashCausalLMBatch":
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        # We assume that if len(requests) == len(self) then the requests are the same
        if len(request_ids) == len(self):
            return self

        device = self.block_tables_tensor.device

        # New values after filtering
        requests_idx_mapping = {}

        # Used to index into tensors
        indices = []

        if not has_triton():
            # slots to keep after filtering
            slot_filtering_indices = torch.zeros(
                self.slots.shape[0], dtype=torch.bool, device=device
            )

        # Create on CPU to only move to GPU once instead of at every copy
        slot_indices = torch.empty(len(request_ids), dtype=torch.int64)
        max_input_length = 0
        max_current_length = 0

        requests = []
        block_tables = []
        all_input_ids = []
        input_ids = []

        prompt_lengths = []
        input_lengths = []
        cache_lengths = []
        prefix_offsets = []
        read_offsets = []
        cu_slots = [0]

        prefilling_mask = []
        prefill_logprob_tokens = []

        stopping_criterias = []
        top_n_tokens = []
        adapter_set = set()

        num_blocks = 0
        max_blocks = 0
        max_slots = 0
        cumulative_slot_tokens = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
            requests_idx_mapping[request_id] = i

            requests.append(self.requests[idx])

            # Prefilling
            request_prefilling = self.prefilling_mask[idx]
            prefilling_mask.append(request_prefilling)

            # Get length
            request_input_length = self.input_lengths[idx]
            request_cache_length = self.cache_lengths[idx]
            max_input_length = max(max_input_length, request_input_length)
            max_current_length = max(
                max_current_length, request_cache_length + request_input_length
            )

            all_input_ids.append(self.all_input_ids[idx])

            prompt_lengths.append(self.prompt_lengths[idx])
            input_lengths.append(request_input_length)
            cache_lengths.append(request_cache_length)
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)

            top_n_tokens.append(self.top_n_tokens[idx])
            prefill_logprob_tokens.append(self.prefill_logprob_tokens[idx])

            ADAPTER_TO_INDEX = get_adapter_to_index()
            adapter_index = ADAPTER_TO_INDEX.get(self.requests[idx].adapter_id, 0)
            adapter_set.add(adapter_index)

            request_block_table = self.block_tables[idx]
            num_blocks += len(request_block_table)
            block_tables.append(request_block_table)

            start_slot = self.cu_slots[idx]
            end_slot = self.cu_slots[idx + 1]
            slot_length = end_slot - start_slot

            if not has_triton():
                # Set slice
                slot_filtering_indices[start_slot:end_slot] = True

            cu_slots.append(cumulative_slot_tokens + slot_length)

            # Input ids if the request was part of a prefilling batch
            # If the batch was decoding we can index into the tensor directly later
            if self.prefilling:
                input_ids.append(self.input_ids[idx])
            else:
                # Copy to tensor (CPU)
                slot_indices[i] = cumulative_slot_tokens + request_cache_length

            cumulative_slot_tokens += slot_length
            max_blocks = max(max_blocks, len(request_block_table))
            max_slots = max(max_slots, slot_length)

        all_input_ids_tensor = self.all_input_ids_tensor[indices]
        block_tables_tensor = self.block_tables_tensor[indices]
        next_token_chooser = self.next_token_chooser.filter(indices)
        top_n_tokens_tensor = self.top_n_tokens_tensor[indices]
        speculative_ids = (
            self.speculative_ids[indices] if self.speculative_ids is not None else None
        )
        prompt_lengths_tensor = self.prompt_lengths_tensor[indices]

        cu_slots = torch.tensor(cu_slots, dtype=torch.int64)

        if not has_triton():
            slots = self.slots[slot_filtering_indices]
        else:
            slots = self.slots.new_empty(cumulative_slot_tokens)
            gpu_cu_slots = cu_slots.to(device)
            slots_indexing_start = self.cu_slots.to(device)[indices]
            slots_filtering(
                max_slots, self.slots, slots, gpu_cu_slots, slots_indexing_start
            )

        if self.prefilling:
            # These values will be set by `FlashCausalLMBatch.prepare_for_prefill`
            position_ids = None
            slot_indices = None
            cache_lengths_tensor = None
            input_lengths_tensor = None
            adapter_meta = None
        else:
            # Index into tensors
            input_ids = self.input_ids[indices]
            position_ids = self.position_ids[indices]
            adapter_indices = self.adapter_meta.adapter_indices[indices]
            input_lengths_tensor = self.input_lengths_tensor[indices]
            cache_lengths_tensor = self.cache_lengths_tensor[indices]

            # Move to GPU now that we have the whole tensor
            slot_indices = slot_indices.to(device)

            adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
            adapter_segments = torch.tensor(
                adapter_segments, dtype=torch.int32, device=device
            )
            adapter_meta = AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            )

        return type(self)(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            slots=slots,
            cu_slots=cu_slots,
            max_input_length=max_input_length,
            max_current_length=max_current_length,
            prefilling=self.prefilling,
            prefilling_mask=prefilling_mask,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            prefill_logprob_tokens=prefill_logprob_tokens,
            prompt_lengths=prompt_lengths,
            prompt_lengths_tensor=prompt_lengths_tensor,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            cache_lengths=cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
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
            adapter_meta=adapter_meta,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        requests_idx_mapping = {}

        prefilling = False
        num_blocks = 0
        total_batch_size = 0
        total_slots = 0
        max_blocks = 0
        max_length = 0
        max_input_length = 0
        max_current_length = 0
        for b in batches:
            total_batch_size += len(b)
            max_blocks = max(max_blocks, b.max_blocks)
            total_slots += len(b.slots)
            num_blocks += b.num_blocks
            speculative_length = (
                b.speculative_ids.shape[1] if b.speculative_ids is not None else 0
            )
            max_input_length = max(max_input_length, b.max_input_length)
            max_current_length = max(max_current_length, b.max_current_length)
            max_length = max(
                max_length,
                max(
                    prompt_length
                    + stopping_criteria.max_new_tokens
                    + speculative_length
                    for prompt_length, stopping_criteria in zip(
                        b.prompt_lengths, b.stopping_criterias
                    )
                ),
            )
            prefilling = prefilling or b.prefilling

        slots = batches[0].slots.new_empty(total_slots)
        cu_slots = torch.zeros(total_batch_size + 1, dtype=torch.int64)
        if prefilling:
            input_ids = []
            # These values will be set by `FlashCausalLMBatch.prepare_for_prefill`
            position_ids = None
            slot_indices = None
            cache_lengths_tensor = None
            input_lengths_tensor = None
            adapter_meta = None
            adapter_segment_builder = None
        else:
            input_ids = batches[0].input_ids.new_empty(total_batch_size)
            position_ids = batches[0].position_ids.new_empty(total_batch_size)
            slot_indices = batches[0].slot_indices.new_empty(total_batch_size)
            input_lengths_tensor = batches[0].input_lengths_tensor.new_empty(
                total_batch_size
            )
            cache_lengths_tensor = batches[0].cache_lengths_tensor.new_empty(
                total_batch_size
            )
            total_indices_size = sum(
                b.adapter_meta.adapter_indices.shape[0] for b in batches
            )
            adapter_indices = batches[0].adapter_meta.adapter_indices.new_empty(
                total_indices_size
            )
            adapter_segment_builder = SegmentConcatBuilder()
            adapter_set = set()

        prompt_lengths_tensor = batches[0].prompt_lengths_tensor.new_empty(
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

        block_tables = []
        cache_lengths = []
        all_input_ids = []

        prompt_lengths = []
        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        prefill_logprob_tokens = []

        next_token_chooser_parameters = []
        fsm_grammar_states = []
        stopping_criterias = []
        top_n_tokens = []
        prefilling_mask = []

        # Cumulative length
        cumulative_batch_size = 0
        cumulative_slots = 0
        cumulative_adapter_indices_size = 0

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

            # Copy tensors (GPU)
            top_n_tokens_tensor[start_index:end_index] = batch.top_n_tokens_tensor
            all_input_ids_tensor[
                start_index:end_index, : batch.all_input_ids_tensor.shape[1]
            ] = batch.all_input_ids_tensor[:, :max_length]

            block_tables_tensor[
                start_index:end_index, : batch.block_tables_tensor.shape[1]
            ] = batch.block_tables_tensor[:, :max_blocks]
            prompt_lengths_tensor[start_index:end_index] = batch.prompt_lengths_tensor

            slots_start_index = cumulative_slots
            slots_end_index = cumulative_slots + len(batch.slots)
            slots[slots_start_index:slots_end_index] = batch.slots
            cu_slots[start_index + 1 : end_index + 1] = (
                batch.cu_slots[1:] + cumulative_slots
            )

            if not prefilling:
                input_ids[start_index:end_index] = batch.input_ids
                position_ids[start_index:end_index] = batch.position_ids
                slot_indices[start_index:end_index] = (
                    batch.slot_indices + cumulative_slots
                )
                input_lengths_tensor[start_index:end_index] = batch.input_lengths_tensor
                cache_lengths_tensor[start_index:end_index] = batch.cache_lengths_tensor

                # Copy over adapter indices
                adapter_start_index = cumulative_adapter_indices_size
                adapter_end_index = (
                    cumulative_adapter_indices_size
                    + batch.adapter_meta.adapter_indices.shape[0]
                )
                adapter_indices[adapter_start_index:adapter_end_index] = (
                    batch.adapter_meta.adapter_indices
                )
                cumulative_adapter_indices_size = adapter_end_index
                adapter_set.update(batch.adapter_meta.adapter_set)
                adapter_segment_builder.concat(
                    batch.adapter_meta.adapter_segments,
                    batch.adapter_meta.segment_indices,
                )
            else:
                if isinstance(batch.input_ids, torch.Tensor):
                    batch.input_ids = batch.input_ids.view(-1, 1).tolist()
                input_ids.extend(batch.input_ids)

            prefilling_mask.extend(batch.prefilling_mask)
            block_tables.extend(batch.block_tables)
            cache_lengths.extend(batch.cache_lengths)
            all_input_ids.extend(batch.all_input_ids)

            prompt_lengths.extend(batch.prompt_lengths)
            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)

            prefill_logprob_tokens.extend(batch.prefill_logprob_tokens)

            next_token_chooser_parameters.extend([r.parameters for r in batch.requests])
            fsm_grammar_states.extend(batch.next_token_chooser.fsm_grammar_states)
            stopping_criterias.extend(batch.stopping_criterias)

            top_n_tokens.extend(batch.top_n_tokens)

            # Update
            cumulative_slots += len(batch.slots)
            cumulative_batch_size += len(batch)

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

        if adapter_segment_builder is not None:
            adapter_segments, adapter_segment_indices = adapter_segment_builder.build()
            adapter_meta = AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            )

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            cache_lengths=cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
            slots=slots,
            cu_slots=cu_slots,
            max_input_length=max_input_length,
            max_current_length=max_current_length,
            prefilling=prefilling,
            prefilling_mask=prefilling_mask,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            prefill_logprob_tokens=prefill_logprob_tokens,
            prompt_lengths=prompt_lengths,
            prompt_lengths_tensor=prompt_lengths_tensor,
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
            adapter_meta=adapter_meta,
        )

    def prepare_for_prefill(self):
        # Prepare values if we need to continue prefilling
        # Speculation must be ignored while we prefill even with chunking
        # it simplifies everything
        assert self.speculative_ids is None

        device = self.block_tables_tensor.device

        if isinstance(self.input_ids, list):
            if len(self) > 1:
                input_ids = np.concatenate(self.input_ids, dtype=np.int64)
            else:
                input_ids = self.input_ids[0]
            self.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)

        self.input_lengths_tensor = torch.tensor(
            self.input_lengths, dtype=torch.int32, device=device
        )
        self.cu_seqlen_prefill = torch.nn.functional.pad(
            torch.cumsum(self.input_lengths_tensor, dim=0), (1, 0)
        ).to(torch.int32)
        self.cache_lengths_tensor = torch.tensor(
            self.cache_lengths, dtype=torch.int32, device=device
        )

        # If the device supports Triton, we can use a fused kernel
        if has_triton():
            self.position_ids = torch.empty(
                len(self.input_ids), dtype=torch.int32, device=device
            )
            self.slot_indices = torch.empty(
                len(self.input_ids), dtype=torch.int64, device=device
            )
            cu_slots_gpu = self.cu_slots.to(device)

            prepare_position_slot_ids(
                self.max_input_length,
                self.cache_lengths_tensor,
                self.cu_seqlen_prefill,
                cu_slots_gpu,
                self.position_ids,
                self.slot_indices,
            )

        sliding_window = get_sliding_windows()
        position_ids = []
        slot_indices = []
        prefill_cache_indices = []
        all_prefill_logprobs = True
        no_prefill_logprobs = True
        prefill_cu_outlens = [0]

        # Cumulative length
        cumulative_length = 0
        cumulative_slot_tokens = 0
        prefill_out_cumulative_length = 0

        adapter_indices_list = []
        adapter_set = set()

        for i, (
            r,
            cache_length,
            input_length,
            prompt_length,
            request_prefilling,
            blocks,
        ) in enumerate(
            zip(
                self.requests,
                self.cache_lengths,
                self.input_lengths,
                self.prompt_lengths,
                self.prefilling_mask,
                self.block_tables,
            )
        ):
            next_chunk_length = input_length

            if not has_triton():
                # Position ids
                request_position_ids = torch.arange(
                    cache_length, cache_length + input_length, dtype=torch.int32
                )
                position_ids.append(request_position_ids)

                if not r.slots:
                    request_slots = [
                        s
                        for b in blocks
                        for s in range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
                    ]
                else:
                    request_slots = r.slots

                request_slot_indices = torch.arange(
                    cache_length + cumulative_slot_tokens,
                    cache_length + cumulative_slot_tokens + input_length,
                    dtype=torch.int64,
                )

                slot_indices.append(request_slot_indices)

                # Update
                cumulative_slot_tokens += len(request_slots)

            # Create tensor to slice into the kv tensor in prefill
            if sliding_window is not None:
                request_prefill_cache_indices = torch.arange(
                    cumulative_length + max(0, input_length - sliding_window),
                    cumulative_length + input_length,
                    dtype=torch.int64,
                )

            # Prefill logprobs is ignored if the request is done prefilling
            prefill_logprobs = r.prefill_logprobs and request_prefilling

            all_prefill_logprobs = all_prefill_logprobs and prefill_logprobs
            no_prefill_logprobs = no_prefill_logprobs and not prefill_logprobs

            if prefill_logprobs:
                prefill_cu_outlens.append(prefill_out_cumulative_length + input_length)
                prefill_out_cumulative_length += input_length
            else:
                prefill_cu_outlens.append(prefill_out_cumulative_length + 1)
                prefill_out_cumulative_length += 1

            if sliding_window is not None:
                prefill_cache_indices.append(request_prefill_cache_indices)

            ADAPTER_TO_INDEX = get_adapter_to_index()
            if ADAPTER_TO_INDEX:
                adapter_index = ADAPTER_TO_INDEX.get(r.adapter_id, 0)
                adapter_indices_list.append(
                    torch.full((next_chunk_length,), adapter_index)
                )
                adapter_set.add(adapter_index)

            # Update
            cumulative_length += next_chunk_length

        if not all_prefill_logprobs and not no_prefill_logprobs:
            prefill_head_indices = []
            prefill_next_token_indices = []

            # Cumulative length
            cumulative_length = 0
            prefill_out_cumulative_length = 0

            for i, (
                r,
                input_length,
                request_prefilling,
            ) in enumerate(
                zip(
                    self.requests,
                    self.input_lengths,
                    self.prefilling_mask,
                )
            ):
                # Prefill logprobs is ignored if the request is done prefilling
                prefill_logprobs = r.prefill_logprobs and request_prefilling

                if prefill_logprobs:
                    prefill_head_indices.append(
                        torch.arange(
                            cumulative_length,
                            cumulative_length + input_length,
                            dtype=torch.int64,
                        )
                    )
                    prefill_next_token_indices.append(
                        prefill_out_cumulative_length + input_length - 1
                    )
                    prefill_out_cumulative_length += input_length
                else:
                    prefill_head_indices.append(
                        torch.tensor(
                            [cumulative_length + input_length - 1],
                            dtype=torch.int64,
                        )
                    )
                    prefill_next_token_indices.append(prefill_out_cumulative_length)
                    prefill_out_cumulative_length += 1

                # Update
                cumulative_length += input_length

        if len(self) > 1:
            if position_ids:
                position_ids = torch.cat(position_ids)
            if slot_indices:
                slot_indices = torch.cat(slot_indices)
            if sliding_window is not None:
                prefill_cache_indices = torch.cat(prefill_cache_indices)
        else:
            if position_ids:
                position_ids = position_ids[0]
            if slot_indices:
                slot_indices = slot_indices[0]
            if sliding_window is not None:
                prefill_cache_indices = prefill_cache_indices[0]

        if not has_triton():
            self.position_ids = position_ids.to(device)
            self.slot_indices = slot_indices.to(device)

        self.prefill_cu_outlens = prefill_cu_outlens
        self.prefill_cache_indices = (
            prefill_cache_indices.to(device) if sliding_window is not None else None
        )

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = self.cu_seqlen_prefill[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = self.cu_seqlen_prefill[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.cat(prefill_head_indices).to(device)
            prefill_next_token_indices = torch.tensor(
                prefill_next_token_indices, dtype=torch.int64, device=device
            )

        self.prefill_head_indices = prefill_head_indices
        self.prefill_next_token_indices = prefill_next_token_indices

        if adapter_set:
            adapter_indices = torch.cat(adapter_indices_list).to(
                dtype=torch.int64, device=device
            )
            adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
        else:
            adapter_indices = torch.zeros_like(self.input_ids)
            adapter_segments = [0, len(adapter_indices)]
            adapter_segment_indices = [len(adapter_indices) - 1]

        adapter_segments = torch.tensor(
            adapter_segments, dtype=torch.int32, device=device
        )

        self.adapter_meta = AdapterBatchMetadata(
            adapter_indices=adapter_indices,
            adapter_set=adapter_set,
            adapter_segments=adapter_segments,
            segment_indices=adapter_segment_indices,
        )

    def __len__(self):
        return len(self.requests)


ADAPTER_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
ROW_PARALLEL = {"o_proj", "down_proj", "lm_head"}


class FlashCausalLM(Model):
    def __init__(
        self,
        model_id: str,
        model_class,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        lora_adapter_ids: Optional[list] = [],
        tokenizer_class: PreTrainedTokenizerBase = AutoTokenizer,
        config_class: PreTrainedTokenizerBase = AutoConfig,
        default_dtype=torch.float16,
        aliases=None,
        # Used for Santacoder override of config
        num_kv_heads: Optional[int] = None,
        # Deepseek V2 uses different QK and V dims.
        head_size: Optional[int] = None,
        skip_special_tokens: bool = True,
        kv_cache_dtype: Optional[torch.dtype] = None,
        support_chunking: bool = True,
    ):
        self.quantize = quantize
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = default_dtype if dtype is None else dtype
        elif SYSTEM == "ipex":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device(f"xpu:{rank}")
                dtype = default_dtype if dtype is None else dtype
            else:
                device = torch.device("cpu")
                dtype = torch.bfloat16 if dtype is None else dtype
                init_cpu_threads_env(rank_id=rank, world_size=world_size)
        else:
            raise NotImplementedError(f"{model_class} is only available on GPU")

        tokenizer = tokenizer_class.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            if isinstance(generation_config.eos_token_id, (list, set)):
                # TODO Huge hack
                tokenizer._eos_token_ids = set(generation_config.eos_token_id)
        except Exception:
            pass

        config = config_class.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.speculator = speculator

        torch.distributed.barrier(group=self.process_group)

        weights_loader = get_loader(quantize, model_id, revision)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
            aliases=aliases,
            weights_loader=weights_loader,
        )

        prefix = ""
        model = model_class(prefix, config, weights)
        torch.distributed.barrier(group=self.process_group)

        # VLM models define the config we care about in their text_config
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            config = text_config

        if getattr(config, "sliding_window", None) is not None:
            set_sliding_window(config.sliding_window)
        else:
            config.sliding_window = None

        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads // self.process_group.size()
        # Validation is done in the model itself
        if num_kv_heads is None:
            num_kv_heads = getattr(config, "num_key_value_heads", None)
            # GPT-2 workaround
            if num_kv_heads is None:
                num_kv_heads = getattr(config, "n_head", None)
        if num_kv_heads is None:
            raise ValueError("Cannot get the number of key/value heads")
        self.num_kv_heads = (
            num_kv_heads // self.process_group.size()
            if num_kv_heads > 1
            else num_kv_heads
        )
        assert self.num_kv_heads > 0

        if head_size is None:
            # Some models use GQA and different sizes for o_proj
            # and q_proj, that allows for that.
            if hasattr(config, "head_dim"):
                self.head_size = config.head_dim
            else:
                self.head_size = config.hidden_size // config.num_attention_heads
        else:
            self.head_size = head_size

        self.cuda_graphs = {}
        self.kv_cache = []
        self.kv_cache_dtype = dtype if kv_cache_dtype is None else kv_cache_dtype

        if ATTENTION == "flashinfer":
            from text_generation_server.layers.attention.flashinfer import (
                create_prefill_state,
                create_decode_state,
                create_prefill_with_paged_kv_state,
            )

            self.prefill_state = create_prefill_state(device=device)
            self.prefill_with_paged_kv_state = create_prefill_with_paged_kv_state(
                device=device
            )

            self.decode_state = create_decode_state(
                device=device,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )

        super().__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=config.sliding_window,
            support_chunking=support_chunking,
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
        self.kv_cache = [
            KVCache(
                num_blocks=num_blocks,
                num_heads=num_heads,
                head_size=head_size,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def cuda_graph_warmup(self, bs: int, max_s: int, max_bt: int):
        input_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
        position_ids = torch.zeros(bs, dtype=torch.int32, device=self.device)
        slots = torch.arange(bs, dtype=torch.int64, device=self.device)
        input_lengths = [max_s] * bs
        cache_lengths = [0] * bs
        input_lengths_tensor = (
            torch.ones(bs, dtype=torch.int32, device=self.device) * max_s
        )
        cache_lengths_tensor = torch.zeros(bs, dtype=torch.int32, device=self.device)
        block_tables = torch.arange(
            max_bt, dtype=torch.int32, device=self.device
        ).repeat(bs)
        block_tables = block_tables.reshape((bs, max_bt))

        if ATTENTION == "flashinfer":
            block_tables = block_tables_to_ragged(
                block_tables=block_tables,
                input_lengths=input_lengths,
                cache_lengths=cache_lengths,
                input_lengths_tensor=input_lengths_tensor,
                cache_lengths_tensor=cache_lengths_tensor,
                max_current_length=max_s,
            )
            from text_generation_server.layers.attention.flashinfer import (
                create_decode_state_cuda_graphs,
            )

            block_tables_ptr = torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            )
            last_page_len = torch.ones(bs, dtype=torch.int32, device=self.device)
            state = create_decode_state_cuda_graphs(
                device=input_ids.device,
                block_tables=block_tables,
                block_tables_ptr=block_tables_ptr,
                last_page_len=last_page_len,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )
        else:
            state = None

        graph = torch.cuda.CUDAGraph()
        self.cuda_graphs[bs] = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "kv_cache": self.kv_cache,
            "block_tables": block_tables,
            "slots": slots,
            "input_lengths": input_lengths_tensor,
            "cache_lengths": cache_lengths_tensor,
            "state": state,
            "graph": graph,
        }

        torch.cuda.synchronize()
        # Run once outside to warmup
        with self._forward_context(
            block_tables=block_tables,
            cu_seqlen_prefill=None,
            input_lengths_tensor=input_lengths_tensor,
            state=state,
            cache_lengths_tensor=cache_lengths_tensor,
        ):
            seqlen = Seqlen(
                input_lengths=input_lengths_tensor,
                cache_lengths=cache_lengths_tensor,
                cu_seqlen_q=None,
                max_q=1,
                max_k=max_s,
            )
            self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=None,
                kv_cache=self.kv_cache,
                block_tables=block_tables,
                slots=slots,
                seqlen=seqlen,
                max_s=max_s,
                prefill_cache_indices=None,
                lm_head_indices=None,
            )
            del seqlen

            torch.cuda.synchronize()

            with torch.cuda.graph(graph, pool=MEM_POOL):
                seqlen = Seqlen(
                    input_lengths=input_lengths_tensor,
                    cache_lengths=cache_lengths_tensor,
                    cu_seqlen_q=None,
                    max_q=1,
                    max_k=max_s,
                )
                logits, speculative_logits = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seqlen_prefill=None,
                    kv_cache=self.kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    seqlen=seqlen,
                    max_s=max_s,
                    prefill_cache_indices=None,
                    lm_head_indices=None,
                )
                self.cuda_graphs[bs]["logits"] = logits
                self.cuda_graphs[bs]["speculative_logits"] = speculative_logits
        torch.cuda.synchronize()

    def warmup(self, batch: FlashCausalLMBatch):
        # The warmup batch is the biggest batch we could ever receive
        self.kv_cache = []
        empty_cache()

        try:
            self.init_kv_cache(
                batch.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.kv_cache_dtype,
                self.device,
            )
            max_bt = batch.max_blocks
            max_s = max_bt * BLOCK_SIZE

            if SYSTEM == "rocm" and os.environ.get("PYTORCH_TUNABLEOP_ENABLED", False):
                torch.cuda.tunable.tuning_enable(False)
            _, batch, _ = self.generate_token(batch)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"Not enough memory to handle {batch.to_pb().current_tokens} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            ) from e

        synchronize(self.device)

        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.kv_cache_dtype).element_size()
        cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
        total_cache_size = self.num_layers * cache_block_size * 2 * dtype_size

        free_memory = get_free_memory(self.device, MEMORY_FRACTION)
        batch_num_blocks = batch.num_blocks if batch is not None else 0

        num_blocks = (
            # Leave 5% for some wiggle room
            int((free_memory * TGI_WIGGLE_ROOM) // total_cache_size)
            # Add batch.num_blocks as we allocated it above, so it is included in the peak memory.
            + batch_num_blocks
        )

        log_master(logger.info, f"KV-cache blocks: {num_blocks}, size: {BLOCK_SIZE}")

        del batch

        self.init_kv_cache(
            num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.kv_cache_dtype,
            self.device,
        )

        if SYSTEM == "rocm":
            if (
                os.environ.get("PYTORCH_TUNABLEOP_ENABLED") is None
                or os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1"
            ):
                torch.cuda.tunable.enable()

                if os.environ.get("PYTORCH_TUNABLEOP_TUNING") != "0":
                    torch.cuda.tunable.tuning_enable(True)

                if os.environ.get("PYTORCH_TUNABLEOP_SEQLENS") is not None:
                    tuning_sequences = [
                        int(val)
                        for val in os.environ["PYTORCH_TUNABLEOP_SEQLENS"].split(",")
                    ]
                elif CUDA_GRAPHS is not None:
                    tuning_sequences = CUDA_GRAPHS
                else:
                    tuning_sequences = [1, 2, 3, 4, 5, 6, 7]

                tunableop_filepath = os.path.join(
                    HUGGINGFACE_HUB_CACHE,
                    f"tunableop_{self.model_id.replace('/', '-')}_tp{self.world_size}_rank{self.rank}.csv",
                )

                log_master(
                    logger.info,
                    f"PyTorch TunableOp is enabled. The warmup may take several minutes, picking the ROCm optimal matrix multiplication kernel for the target lengths {', '.join([str(seqlen) for seqlen in tuning_sequences])}, with typical 5-8% latency improvement for small sequence lengths. The picked GEMMs are saved in the file {tunableop_filepath}. To disable TunableOp, please launch TGI with `PYTORCH_TUNABLEOP_ENABLED=0`.",
                )

                torch.cuda.tunable.set_filename(
                    tunableop_filepath, insert_device_ordinal=False
                )

                if os.path.isfile(tunableop_filepath):
                    log_master(
                        logger.info,
                        f"The file {tunableop_filepath} already exists and will be reused.",
                    )
                    torch.cuda.tunable.read_file(tunableop_filepath)

                os.makedirs(HUGGINGFACE_HUB_CACHE, exist_ok=True)

                for seqlen in tuning_sequences:
                    log_master(logger.info, f"Warming up TunableOp for seqlen={seqlen}")
                    self.tunableop_warmup(seqlen)
                    torch.cuda.tunable.write_file(tunableop_filepath)
                if os.environ.get("PYTORCH_TUNABLEOP_TUNING_AFTER_WARMUP") != "1":
                    torch.cuda.tunable.tuning_enable(False)
            else:
                log_master(
                    logger.info,
                    "PyTorch ROCm TunableOp (https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable) is disabled. TunableOp brings an additional 5-8% latency improvement for small sequence lengths but requires a warmup. If necessary, please use the environment variable PYTORCH_TUNABLEOP_ENABLED=1 to enable TunableOp.",
                )

        if CUDA_GRAPHS:
            try:
                log_master(
                    logger.info, f"Cuda Graphs are enabled for sizes {CUDA_GRAPHS}"
                )
                # Warmup cuda graphs
                for bs in CUDA_GRAPHS:
                    if self.speculate is None or self.speculate + 1 <= bs:
                        self.cuda_graph_warmup(bs, max_s, max_bt)
            except torch.cuda.OutOfMemoryError:
                logger.exception("Decode cuda graph warmup failed")
        else:
            log_master(
                logger.info, f"Cuda Graphs are disabled (CUDA_GRAPHS={CUDA_GRAPHS})."
            )

        return int(num_blocks * BLOCK_SIZE)

    def tunableop_warmup(self, seqlen: int):
        input_ids = torch.zeros(seqlen, dtype=torch.int64, device=self.device)
        position_ids = torch.zeros(seqlen, dtype=torch.int32, device=self.device)
        slots = torch.arange(seqlen, dtype=torch.int64, device=self.device)

        # Dummy value, some models (starcoder2) don't accept `None`.
        input_lengths = torch.ones(seqlen, dtype=torch.int32, device=self.device)
        cache_lengths_tensor = torch.zeros(
            seqlen, dtype=torch.int32, device=self.device
        )
        cu_seqlen_prefill = torch.tensor(
            [0, seqlen], device=self.device, dtype=torch.int32
        )
        max_s = seqlen
        seqlen = Seqlen(
            input_lengths=input_lengths,
            cache_lengths=cache_lengths_tensor,
            cu_seqlen_q=cu_seqlen_prefill,
            max_q=1,
            max_k=seqlen,
        )

        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=self.kv_cache,
            block_tables=None,
            seqlen=seqlen,
            slots=slots,
            max_s=max_s,
            lm_head_indices=None,
            prefill_cache_indices=None,
        )

    def forward(
        self, batch: FlashCausalLMBatch, adapter_data: AdapterBatchData
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

        bs = input_ids.shape[0]
        sorted_padded_bs = sorted([k for k in self.cuda_graphs.keys() if k >= bs])
        if sorted_padded_bs:
            # Get associated cuda graph
            cuda_graph = self.cuda_graphs[sorted_padded_bs[0]]
        else:
            cuda_graph = None

        if cu_seqlen_prefill is not None or cuda_graph is None:
            if ATTENTION == "flashinfer":
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=batch.input_lengths,
                    cache_lengths=batch.cache_lengths,
                    input_lengths_tensor=batch.input_lengths_tensor,
                    cache_lengths_tensor=batch.cache_lengths_tensor,
                    max_current_length=batch.max_current_length,
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
                    adapter_data=adapter_data,
                )
                if batch.prefill_cache_indices is not None:
                    batch.prefill_cache_indices = None
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
                input_lengths_tensor=batch.input_lengths_tensor,
                cache_lengths_tensor=batch.cache_lengths_tensor,
                max_current_length=batch.max_current_length,
            )
            # assert block_tables.shape[0] >= slots.shape[0]
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

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch], Tuple[int, int]]:
        start = time.time_ns()
        prefill = batch.prefilling
        if prefill:
            batch.prepare_for_prefill()

        prefill_logprobs = batch.prefill_next_token_indices is not None

        # Update adapter indices for speculative tokens (if present)
        adapter_meta = batch.adapter_meta
        if batch.speculative_ids is not None:
            B, speculative_length = batch.speculative_ids.shape
            new_length = speculative_length + 1
            adapter_indices = (
                adapter_meta.adapter_indices.unsqueeze(-1)
                .expand(B, new_length)
                .reshape(-1)
            )
            adapter_segments = adapter_meta.adapter_segments * new_length
            adapter_meta = AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_meta.adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_meta.segment_indices,
            )

        # Assign pointers to adapter weights
        # TODO(travis): don't update this if indices haven't changed
        adapter_data = AdapterBatchData.from_meta(
            adapter_meta,
            self.layer_to_adapter_weights,
            prefill,
            batch.prefill_head_indices,
        )

        out, speculative_logits = self.forward(batch, adapter_data)

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
            if len(batch) > 1 and prefill_logprobs:
                # We create the prefill_tokens_indices tensor that will be used to gather prefill logprobs
                # When batch == 1, we will just use the batch.input_ids values directly
                prefill_tokens_indices = batch.input_ids.new_zeros(len(out))
        else:
            prefill_logprobs = None
            next_token_logits = out

        finished_prefilling = True
        next_chunk_lengths = []
        current_prefilling_mask = batch.prefilling_mask
        if prefill:
            if get_support_chunking():
                next_prefilling_mask = []
                # Budget in tokens for the next batch
                # We remove (len(batch) - 1) to always have enough space for at least a single decode
                # for the remaining requests -1 because the first request does not need to be removed from the budget
                # (ex: you have one request in the batch, you want it to take the full budget not budget -1)
                batch_budget = get_max_prefill_tokens() - (len(batch) - 1)
                # We reverse to prioritize older requests
                # zip() is not reversible so reverse the underlying lists instead
                for cache_length, input_length, prompt_length in zip(
                    reversed(batch.cache_lengths),
                    reversed(batch.input_lengths),
                    reversed(batch.prompt_lengths),
                ):
                    remaining_prefill_tokens = max(
                        prompt_length - cache_length - input_length, 0
                    )
                    if remaining_prefill_tokens > 0:
                        next_chunk_length = max(
                            min(remaining_prefill_tokens, batch_budget), 1
                        )
                        batch_budget -= next_chunk_length
                        finished_prefilling = False
                        next_prefilling_mask.append(True)
                    else:
                        # FIXME: use true number of accepted tokens instead of 1
                        # Since speculation will be turned off, this is always true
                        next_chunk_length = 1
                        next_prefilling_mask.append(False)
                    next_chunk_lengths.append(next_chunk_length)

                # Reverse back the obtained values²
                next_chunk_lengths.reverse()
                next_prefilling_mask.reverse()
            else:
                # The model does not support chunking
                # We know we only do a single prefill
                finished_prefilling = True
                next_prefilling_mask = [False] * len(batch)

            batch.prefilling = not finished_prefilling
            batch.prefilling_mask = next_prefilling_mask

        speculate = get_speculate()
        (
            next_input_ids,
            next_token_logprobs,
            logprobs,
            accepted_ids,
            speculative_ids,
        ) = batch.next_token_chooser(
            batch.all_input_ids_tensor[:, : batch.max_current_length],
            next_token_logits,
            speculate,
            batch.speculative_ids,
            speculative_logits,
        )

        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens, batch.top_n_tokens_tensor, logprobs, accepted_ids
        )

        # Since we are done prefilling, all the tensors that were concatenating values for all the requests
        # instantly become of shape [BATCH_SIZE]
        if prefill and finished_prefilling:
            indices = batch.cu_seqlen_prefill[1:] - 1
            batch.position_ids = batch.position_ids[indices]
            batch.slot_indices = batch.slot_indices[indices]
            batch.adapter_meta.adapter_indices = batch.adapter_meta.adapter_indices[
                indices
            ]

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.prompt_lengths,
            batch.cache_lengths,
            batch.input_lengths,
            batch.all_input_ids,
            accepted_ids,
            current_prefilling_mask,
            batch.prefilling_mask,
        )

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        # Cumulative length
        cu_accepted_ids = torch.nn.functional.pad(
            torch.cumsum(accepted_ids, dim=0), (1, 0)
        )
        cumulative_length = 0
        for i, (
            request,
            prompt_length,
            cache_length,
            input_length,
            all_input_ids,
            n_accepted_ids,
            request_was_prefilling,
            request_is_prefilling,
        ) in enumerate(iterator):
            # Used to gather prefill logprobs
            # Copy batch.all_input_ids_tensor to prefill_token_indices
            if request.prefill_logprobs and request_was_prefilling:
                # Indexing metadata
                out_start_index = batch.prefill_cu_outlens[i]
                out_end_index = batch.prefill_cu_outlens[i + 1]

                # Logprobs generated by the model are for the next token
                # So we need to translate the id tensor by 1
                ids = batch.all_input_ids_tensor[
                    i, cache_length + 1 : cache_length + input_length + 1
                ]
                if len(batch) > 1:
                    prefill_tokens_indices[out_start_index:out_end_index] = ids
                else:
                    # Set prefill_tokens_indices to the correct slice
                    prefill_tokens_indices = ids

            # If the device does not support triton, we copy one by one
            if not request_is_prefilling and not has_triton():
                # Only save tokens if we are done prefilling for this request
                batch.all_input_ids_tensor[
                    i,
                    batch.cache_lengths_tensor[i]
                    + batch.input_lengths[i] : batch.cache_lengths_tensor[i]
                    + batch.input_lengths[i]
                    + accepted_ids[i],
                ] = next_input_ids[cu_accepted_ids[i] : cu_accepted_ids[i + 1]]
            cumulative_length += input_length

        # If the device support triton, we can use a fused kernel
        if has_triton():
            copy_next_input_ids_inplace(
                speculate + 1,
                batch.all_input_ids_tensor,
                batch.cache_lengths_tensor,
                batch.input_lengths_tensor,
                batch.prompt_lengths_tensor,
                next_input_ids,
                cu_accepted_ids,
            )

        # Update values
        # These values can be updated without a GPU -> CPU sync
        if not prefill or (prefill and finished_prefilling):
            batch.input_ids = next_input_ids[cu_accepted_ids[1:] - 1]
            batch.speculative_ids = speculative_ids
            batch.position_ids += accepted_ids
            batch.cache_lengths_tensor += batch.input_lengths_tensor + accepted_ids - 1
            batch.input_lengths_tensor = torch.ones_like(batch.input_lengths_tensor)
            batch.slot_indices += accepted_ids

        if prefill and prefill_logprobs:
            # Get prefill logprobs with inplace softmax (avoid copying the `out` tensor (max_batch_prefill_tokens * vocab_size))
            torch.log_softmax(out, -1, out=out)
            prefill_logprobs_tensor = out
            prefill_logprobs = torch.gather(
                prefill_logprobs_tensor, 1, prefill_tokens_indices.view(-1, 1)
            )
            # GPU <-> CPU sync
            prefill_logprobs = prefill_logprobs.view(-1).tolist()

        # Does a GPU <-> CPU sync internally
        if prefill and finished_prefilling:
            # adjust segment lengths to account for all request lengths being 1 during decoding
            adapter_segments, _ = find_segments(batch.adapter_meta.adapter_indices)
            batch.adapter_meta.adapter_segments = torch.tensor(
                adapter_segments,
                dtype=torch.int32,
                device=batch.adapter_meta.adapter_segments.device,
            )

        # GPU <-> CPU sync
        next_token_logprobs = next_token_logprobs.tolist()
        next_token_ids = next_input_ids.tolist()
        accepted_ids = accepted_ids.tolist()

        # Update values if we need to continue prefilling
        # This represents the `else` case of the `Update values` if above
        # but since this require the `next_token_ids` to be on CPU, it is better to do it here
        if prefill and not finished_prefilling:
            # Speculation must be ignored while we prefill even with chunking
            # it simplifies everything
            assert batch.speculative_ids is None

            all_postfix_ids = []
            for i, (
                request_prefilling,
                next_token_id,
                all_input_ids,
                cache_length,
                input_length,
                next_chunk_length,
            ) in enumerate(
                zip(
                    batch.prefilling_mask,
                    next_token_ids,
                    batch.all_input_ids,
                    batch.cache_lengths,
                    batch.input_lengths,
                    next_chunk_lengths,
                )
            ):
                if request_prefilling:
                    next_cache_length = cache_length + input_length
                    # Get new prompt IDs to prefill
                    postfix_ids = all_input_ids[
                        next_cache_length : next_cache_length + next_chunk_length
                    ]
                else:
                    # This request is done prefilling, the new id is the one selected the sampling method
                    postfix_ids = [next_token_id]

                all_postfix_ids.append(postfix_ids)

            batch.input_ids = all_postfix_ids

        start_decode = time.time_ns()

        # Results
        generations: List[Generation] = []
        stopped = True

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.prompt_lengths,
            batch.cache_lengths,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.next_token_chooser.do_sample,
            batch.next_token_chooser.seeds,
            batch.top_n_tokens,
            current_prefilling_mask,
            batch.prefilling_mask,
            accepted_ids,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        # Reset max_input_length
        batch.max_input_length = 0
        # For each member of the batch
        index = 0
        for i, (
            request,
            prompt_length,
            cache_length,
            input_length,
            prefix_offset,
            read_offset,
            stopping_criteria,
            all_input_ids,
            do_sample,
            seed,
            top_n_tokens,
            request_was_prefilling,
            request_is_prefilling,
            n_accepted_ids,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Compute logprobs first as, even though we might skip the token,
            # it can still be required to compute the logprobs
            # modulo on request.id as it is robust to batch.filter whereas the index in the batch is not and we need
            # this state to be stable
            if request.id % self.world_size == self.rank:
                # Prefill
                if request_was_prefilling and request.prefill_logprobs:
                    out_start_index = batch.prefill_cu_outlens[i]
                    out_end_index = batch.prefill_cu_outlens[i + 1]
                    if not request_is_prefilling:
                        # The request is dones prefilling, meaning that we started generating new tokens
                        # The last logprob is a logprob for a generated token that was not part of the prompt
                        # We need to remove it
                        out_end_index -= 1

                    request_prefill_logprobs = prefill_logprobs[
                        out_start_index:out_end_index
                    ]
                    # Logprobs generated by the model are for the next token
                    # So we need to translate the id tensor by 1
                    prefill_token_ids = all_input_ids[
                        cache_length + 1 : cache_length + input_length + 1
                    ]

                    past_prefill_logprob_tokens = batch.prefill_logprob_tokens[i]

                    if past_prefill_logprob_tokens is None:
                        # add nan for cached prompt tokens/first token
                        request_prefill_logprobs = [float("nan")] * (
                            cache_length + 1
                        ) + request_prefill_logprobs
                        prefill_token_ids = (
                            all_input_ids[: cache_length + 1] + prefill_token_ids
                        )

                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )

                    prefill_logprob_tokens = Tokens(
                        prefill_token_ids,
                        request_prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                    if past_prefill_logprob_tokens is not None:
                        prefill_logprob_tokens = (
                            past_prefill_logprob_tokens + prefill_logprob_tokens
                        )

                    batch.prefill_logprob_tokens[i] = prefill_logprob_tokens
                else:
                    batch.prefill_logprob_tokens[i] = None

            # If it is, the tokens we decoded should be ignored
            if request_is_prefilling:
                # Make sure that we do not stop as even though this request did not create a token, it is still
                # processing
                stopped = False
                new_input_length = next_chunk_lengths[i]
                new_cache_length = cache_length + input_length
            else:
                new_input_length = 1
                new_cache_length = cache_length + input_length + n_accepted_ids - 1
                # Append next token to all tokens
                next_token_texts = []
                left = 0

                if n_accepted_ids > 1:
                    log_master(logger.debug, f"speculated ids {n_accepted_ids - 1}")

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

                # Shard generations
                # All generations will be appended in the rust sharded client
                if request.id % self.world_size == self.rank:
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
                        batch.prefill_logprob_tokens[i],
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
                        batch.next_token_chooser.advance_grammar_single(
                            i, next_token_id
                        )
                    )

            # Update values
            index += n_accepted_ids
            batch.cache_lengths[i] = new_cache_length
            batch.max_input_length = max(batch.max_input_length, new_input_length)
            batch.input_lengths[i] = new_input_length
            current_length = new_cache_length + new_input_length
            batch.max_current_length = max(batch.max_current_length, current_length)

            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids

        if stopped:
            # No need to return a batch if we know that all requests stopped
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        if prefill and finished_prefilling:
            # We do not need prefill tensors anymore
            batch.cu_seqlen_prefill = None
            batch.prefill_cache_indices = None
            batch.prefill_cu_outlens = None
            batch.prefill_head_indices = None
            batch.prefill_next_token_indices = None

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)

    def _forward_context(
        self,
        *,
        block_tables: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        input_lengths_tensor: torch.Tensor,
        cache_lengths_tensor: torch.Tensor,
        state: Optional[Any] = None,
    ) -> ContextManager:
        if ATTENTION != "flashinfer":
            return nullcontext()

        from text_generation_server.layers.attention.flashinfer import (
            use_decode_state,
            use_prefill_with_paged_kv_state,
        )

        if cu_seqlen_prefill is not None:
            return use_prefill_with_paged_kv_state(
                state=(
                    state if state is not None else self.prefill_with_paged_kv_state
                ),
                block_tables=block_tables,
                cu_seqlens=cu_seqlen_prefill,
                input_lengths=input_lengths_tensor + cache_lengths_tensor,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                page_size=BLOCK_SIZE,
                dtype=self.dtype,
                window_left=self.sliding_window,
            )
        else:
            assert input_lengths_tensor is not None
            return use_decode_state(
                state=state if state is not None else self.decode_state,
                input_lengths=input_lengths_tensor + cache_lengths_tensor,
                block_tables=block_tables,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                page_size=BLOCK_SIZE,
                kv_cache_dtype=self.kv_cache_dtype,
                dtype=self.dtype,
                window_left=self.sliding_window,
            )
