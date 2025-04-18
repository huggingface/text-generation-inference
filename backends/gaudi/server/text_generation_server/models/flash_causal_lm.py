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
    Iterable,
    Optional,
    Tuple,
    List,
    Type,
    Dict,
    Union,
)
import torch.nn.functional as F
from text_generation_server.adapters import AdapterBatchData, AdapterBatchMetadata
from text_generation_server.utils.chunks import concat_text_chunks
from text_generation_server.models import Model
from text_generation_server.utils.log import log_master
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
    BLOCK_SIZE,
    REQUEST_LOGPROBS,
    TGI_WIGGLE_ROOM,
    get_adapter_to_index,
)
from text_generation_server.layers.attention import (
    KVCache,
    Seqlen,
    HPUPagedAttentionMetadata,
    trim_attn_metadata,
    trim_seqlen_metadata,
)
from text_generation_server.utils import StoppingCriteria, HeterogeneousNextTokenChooser
from text_generation_server.utils.dist import MEMORY_FRACTION
from text_generation_server.utils.quantization import get_loader
from text_generation_server.utils.segments import SegmentConcatBuilder, find_segments
from text_generation_server.utils.import_utils import (
    empty_cache,
    synchronize,
    get_free_memory,
)

import vllm_hpu_extension.environment as environment
import habana_frameworks.torch as htorch
import itertools
from vllm_hpu_extension.ops import batch2block, block2batch

tracer = trace.get_tracer(__name__)

# Will be set in init
SLIDING_WINDOW: Optional[int] = None


def set_sliding_window(sliding_window: int):
    global SLIDING_WINDOW
    SLIDING_WINDOW = sliding_window


def get_sliding_windows() -> int:
    global SLIDING_WINDOW
    return SLIDING_WINDOW


def prepare_for_decode(
    dtype, use_contiguous_pa, device, slot, block_tables, batch_size
):
    # Prepare values if we need to continue decoding
    # need for HPUPagedAttentionMetadata preparation
    def flatten(in_list):
        return list(itertools.chain(*in_list))

    def gather_list(input, indices, v):
        return [input[i] if i is not None else v for i in indices]

    def pad_list(input, k, v):
        input_len = len(input)
        target_len = (input_len + k - 1) // k * k
        padding = target_len - input_len
        return input + [v] * padding

    last_block_usage = slot % BLOCK_SIZE + 1
    block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
    block_usage = [
        [BLOCK_SIZE] * (len(bt) - 1) + [lbu]
        for bt, lbu in zip(block_tables, last_block_usage)
        if bt
    ]

    block_list = flatten(block_tables)
    block_groups = flatten(block_groups)
    block_usage = flatten(block_usage)
    assert len(block_list) == len(block_groups)
    assert len(block_list) == len(block_usage)
    if use_contiguous_pa:
        block_bucket_size = max(max(block_list) + 1, len(block_list))
        #     block_bucket_size = self.bucketing_ctx.get_padded_decode_num_blocks(
        #         block_bucket_size)
        indices: List[Any]
        indices = [None] * block_bucket_size
        for i, bid in enumerate(block_list):
            indices[bid] = i
        block_list = gather_list(block_list, indices, 0)
        block_groups = gather_list(block_groups, indices, -1)
        block_usage = gather_list(block_usage, indices, 1)
    else:
        block_bucket_size = len(block_list)
        block_list = pad_list(block_list, block_bucket_size, 0)
        block_groups = pad_list(block_groups, block_bucket_size, -1)
        block_usage = pad_list(block_usage, block_bucket_size, 1)

    block_list = torch.tensor(block_list, dtype=torch.int, device=device)
    block_groups = torch.tensor(block_groups, dtype=torch.int, device=device)
    block_usage = torch.tensor(block_usage, dtype=dtype, device=device)
    block_mapping = torch.nn.functional.one_hot(block_groups, num_classes=batch_size)
    mask = torch.arange(0, BLOCK_SIZE, device=device, dtype=torch.int32).unsqueeze(0)
    mask = mask >= block_usage.unsqueeze(-1)
    attn_bias = torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf)
    ones = torch.ones(
        (block_mapping.size(0),), device=device, dtype=block_mapping.dtype
    )
    sums = batch2block(block2batch(ones, block_mapping), block_mapping)
    block_scales = torch.reciprocal(torch.maximum(ones, sums))
    return trim_attn_metadata(
        HPUPagedAttentionMetadata(
            block_list=block_list,
            block_groups=block_groups,
            block_usage=block_usage,
            block_mapping=block_mapping.to(dtype),
            attn_bias=attn_bias,
            block_scales=block_scales,
        )
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

    hpu_attn_meta: Optional[HPUPagedAttentionMetadata]

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
            ### XXX: This consumes so much memory on long requests
            ### Deactivating it by default seems like the best course.
            if not REQUEST_LOGPROBS:
                r.prefill_logprobs = False
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

        for i, request_blocks in enumerate(block_tables):
            block_tables_tensor[i, : len(request_blocks)] = torch.tensor(request_blocks)

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
            hpu_attn_meta=None,
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

        slots = self.slots[slot_filtering_indices]

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
            hpu_attn_meta=None,
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
            if (
                batches[0].position_ids is not None
                and batches[0].position_ids.dim() == 2
            ):
                # Qwen2_vl case:
                position_ids = batches[0].position_ids.new_empty(
                    (total_batch_size, batches[0].position_ids.shape[-1])
                )
            else:
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

        # We skip computing the speculative_ids when the batch size is too large, so
        # we must check that all batches have them, otherwise they must be discarded
        if get_speculate() > 0 and all(b.speculative_ids is not None for b in batches):
            speculative_ids = torch.cat([b.speculative_ids for b in batches], dim=0)
        else:
            speculative_ids = None

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
            hpu_attn_meta=None,
        )

    def prepare_for_decode(self, dtype, use_contiguous_pa):
        block_num = self.cache_lengths_tensor // BLOCK_SIZE + 1
        block_tables = []
        for i, bt in enumerate(self.block_tables):
            block_tables.append(bt[0 : block_num[i]])

        self.hpu_attn_meta = prepare_for_decode(
            dtype,
            use_contiguous_pa,
            self.block_tables_tensor.device,
            self.slots[self.slot_indices],
            block_tables,
            self.input_ids.size(0),
        )

    def prepare_for_prefill(self):
        # Prepare values if we need to continue prefilling
        # Speculation must be ignored while we prefill even with chunking
        # it simplifies everything
        assert self.speculative_ids is None

        device = self.block_tables_tensor.device

        # hpu does not support varlen for prefill, use sdpa instead. so need to pad input_tensor, position
        # padding to left to work with sliding window
        # use prefill_cache_indices to indicate the valid kv slot, update prefill_next_token_indices to indicate
        # the right logit position
        input_ids_padded_length = []
        # need extra pad to match warmup seq
        extra_pad = 0
        if isinstance(self.input_ids, list) and len(self) > 1:
            input_ids_padded_length = []
            input_ids = []
            for input_id in self.input_ids:
                padded = self.max_input_length - len(input_id) + extra_pad
                if padded > 0:
                    input_id = [0] * padded + input_id
                input_ids.append(input_id)
                input_ids_padded_length.append(padded)
            input_ids = np.concatenate(input_ids, dtype=np.int64)
            self.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        elif isinstance(self.input_ids, list):
            input_ids = self.input_ids[0]
            input_ids_padded_length.append(extra_pad)
            input_ids = [0] * extra_pad + input_ids
            self.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        else:
            self.input_ids = F.pad(self.input_ids, (extra_pad, 0), value=0)
            input_ids_padded_length.append(extra_pad)

        self.input_lengths_tensor = torch.tensor(
            self.input_lengths, dtype=torch.int32, device=device
        )
        cu_seqlen_prefill = self.input_lengths_tensor.new_zeros(len(self) + 1)
        torch.cumsum(self.input_lengths_tensor, out=cu_seqlen_prefill[1:], dim=0)
        self.cu_seqlen_prefill = cu_seqlen_prefill.to(torch.int32)
        self.cache_lengths_tensor = torch.tensor(
            self.cache_lengths, dtype=torch.int32, device=device
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

            # Position ids
            request_position_ids = torch.arange(
                cache_length, cache_length + input_length, dtype=torch.int32
            )
            request_position_ids = F.pad(
                request_position_ids, (input_ids_padded_length[i], 0), value=1
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
            # hpu need request_prefill_cache_indices to skip padding in kv cache
            sliding_window = get_sliding_windows()
            if sliding_window is None:
                sliding_window = input_length
            cumulative_length += input_ids_padded_length[i]
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
            prefill_cache_indices = torch.cat(prefill_cache_indices)
        else:
            if position_ids:
                position_ids = position_ids[0]
            if slot_indices:
                slot_indices = slot_indices[0]
            prefill_cache_indices = prefill_cache_indices[0]

        self.position_ids = position_ids.to(device)
        self.slot_indices = slot_indices.to(device)

        self.prefill_cu_outlens = prefill_cu_outlens
        self.prefill_cache_indices = torch.zeros_like(self.input_ids, dtype=torch.bool)
        self.prefill_cache_indices[prefill_cache_indices.to(device)] = True

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
        input_ids_padded_length_tensor = torch.cumsum(
            torch.tensor(input_ids_padded_length, dtype=torch.int64, device=device),
            dim=-1,
        )
        if self.prefill_head_indices is not None:
            self.prefill_head_indices = (
                self.prefill_head_indices + input_ids_padded_length_tensor
            )

        if self.prefill_next_token_indices is not None:
            self.prefill_next_token_indices = (
                self.prefill_next_token_indices + input_ids_padded_length_tensor
            )

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

        device = torch.device("hpu")
        dtype = torch.bfloat16 if dtype is None else dtype

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

        prefix = None
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
        self.config = config
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

        if htorch.utils.internal.is_lazy():
            htorch.hpu.wrap_in_hpu_graph(model, disable_tensor_cache=False)
        environment.set_model_config(self.config)
        self.use_contiguous_pa = (
            os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() == "true"
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

    def warmup(
        self,
        batch: FlashCausalLMBatch,
        max_input_tokens: Optional[int],
        max_total_tokens: Optional[int],
    ):
        # The warmup batch is the biggest batch we could ever receive
        self.kv_cache = []
        empty_cache()

        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.kv_cache_dtype).element_size()
        cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
        total_cache_size = self.num_layers * cache_block_size * 2 * dtype_size

        try:
            self.init_kv_cache(
                batch.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.kv_cache_dtype,
                self.device,
            )

            batch_num_blocks = batch.num_blocks

            num_tokens = batch.to_pb().current_tokens
            synchronize(self.device)
            free_memory = get_free_memory(
                self.device, MEMORY_FRACTION * TGI_WIGGLE_ROOM
            )
            real_free_memory = get_free_memory(self.device, MEMORY_FRACTION)
            log_master(
                logger.debug,
                f"Free memory {free_memory / 1e9:.2f}GB , (real: {real_free_memory / 1e9:.2f}GB",
            )

            _, _batch, _ = self.generate_token([batch])
        except Exception:
            raise RuntimeError(
                f"Not enough memory to handle {num_tokens} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            )

        synchronize(self.device)
        free_memory = get_free_memory(self.device, MEMORY_FRACTION * TGI_WIGGLE_ROOM)
        kv_memory = free_memory
        num_blocks = (
            # Leave 5% for some wiggle room
            int(kv_memory // total_cache_size)
            # Add batch.num_blocks as we allocated it above, so it is included in the peak memory.
            + batch_num_blocks
        )

        log_master(logger.info, f"KV-cache blocks: {num_blocks}, size: {BLOCK_SIZE}")
        if max_total_tokens is None:
            max_total_tokens = sum(batch.cache_lengths)

        if max_input_tokens is None:
            max_input_tokens = max_total_tokens - 1

        del _batch, batch
        self.kv_cache = []
        empty_cache()

        self.init_kv_cache(
            num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.kv_cache_dtype,
            self.device,
        )
        return int(num_blocks * BLOCK_SIZE), max_input_tokens, max_total_tokens

    def warmup_prefill(self, prompt_len: int, bs: int):
        input_ids = torch.zeros(
            prompt_len, dtype=torch.int64, device=self.device
        ).repeat(bs)
        position_ids = torch.arange(
            prompt_len, dtype=torch.int32, device=self.device
        ).repeat(bs)
        max_bt = (prompt_len // BLOCK_SIZE + 1) * bs
        block_tables = torch.arange(
            max_bt, dtype=torch.int32, device=self.device
        ).reshape(bs, -1)
        slot_acc = []
        for i in range(bs):
            slots = []
            for b in block_tables[i]:
                slots.extend(range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE))
            slot_acc.extend(slots[:prompt_len])
        slots = torch.tensor(slot_acc, dtype=torch.int64, device=self.device)

        input_lengths = (
            torch.ones(bs, dtype=torch.int32, device=self.device) * prompt_len
        )
        cache_lengths_tensor = torch.zeros(bs, dtype=torch.int32, device=self.device)
        cu_seqlen_prefill = torch.zeros(bs + 1, device=self.device, dtype=torch.int32)
        torch.cumsum(input_lengths, -1, out=cu_seqlen_prefill[1:])

        seqlen = Seqlen(
            input_lengths=input_lengths,
            cache_lengths=cache_lengths_tensor,
            cu_seqlen_q=cu_seqlen_prefill,
        )
        lm_head_indices = input_lengths - 1

        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=self.kv_cache,
            slots=slots,
            seqlen=trim_seqlen_metadata(seqlen),
            lm_head_indices=lm_head_indices,
            adapter_data=None,
            hpu_attention_meta=None,
        )

    def warmup_decode(self, bs: int, block_num: int):
        input_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
        position_ids = torch.arange(bs, dtype=torch.int32, device=self.device)
        block_tables = torch.arange(
            start=1, end=block_num + 1, dtype=torch.int32, device=self.device
        ).reshape(bs, -1)
        slots = []
        past_len = (
            len(block_tables[0]) * BLOCK_SIZE - 1
        )  # for decode, we only need to pass the past token
        # fetch the last blocked to warmup block num
        for i in range(bs):
            slots.append(BLOCK_SIZE * block_tables[i][-1] + BLOCK_SIZE - 1)
        slots = torch.tensor(slots, dtype=torch.int64, device=self.device)
        input_lengths = torch.ones(bs, dtype=torch.int32, device=self.device)
        cache_lengths_tensor = (
            torch.ones(bs, dtype=torch.int32, device=self.device) * past_len
        )
        cu_seqlen_prefill = torch.zeros(bs + 1, device=self.device, dtype=torch.int32)
        torch.cumsum(input_lengths, -1, out=cu_seqlen_prefill[1:])

        seqlen = Seqlen(
            input_lengths=input_lengths,
            cache_lengths=cache_lengths_tensor,
            cu_seqlen_q=cu_seqlen_prefill,
        )
        block_num = cache_lengths_tensor // BLOCK_SIZE + 1
        block_tables_valid = []
        for i, bt in enumerate(block_tables.tolist()):
            block_tables_valid.append(bt[0 : block_num[i]])

        hpu_attention_meta = prepare_for_decode(
            self.dtype,
            self.use_contiguous_pa,
            self.device,
            slots,
            block_tables_valid,
            bs,
        )

        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            kv_cache=self.kv_cache,
            slots=slots,
            seqlen=trim_seqlen_metadata(seqlen),
            lm_head_indices=None,
            adapter_data=None,
            hpu_attention_meta=hpu_attention_meta,
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

            # Slots can be discontiguous when prefix caching is enabled, so we need to expand the slot_indices,
            # then update the slots with the additional indices to ensure we're grabbing the ones that have been
            # allocated
            slot_indices = (
                batch.slot_indices.unsqueeze(-1).expand(B, new_length) + arange_int
            ).view(-1)
            slots = batch.slots[slot_indices]

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

        seqlen = Seqlen(
            input_lengths=input_lengths,
            cache_lengths=cache_lengths_tensor,
            cu_seqlen_q=cu_seqlen_prefill,
        )
        kwargs = {}
        if htorch.utils.internal.is_lazy():
            kwargs["bypass_hpu_graphs"] = False
        if batch.prefill_cache_indices is not None:
            slots_pad = torch.zeros_like(input_ids)
            slots_pad[batch.prefill_cache_indices] = slots
            slots = slots_pad
        logits, speculative_logits = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            slots=slots,
            seqlen=trim_seqlen_metadata(seqlen),
            lm_head_indices=lm_head_indices,
            # TODO not support adapter now, need the add in the future
            adapter_data=None,
            hpu_attention_meta=batch.hpu_attn_meta,
            **kwargs,
        )
        return logits, speculative_logits

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batches: List[FlashCausalLMBatch]
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch], Tuple[int, int]]:
        if len(batches) > 1:
            batch = self.batch_type.concatenate(batches)
        else:
            batch = batches[0]
        start = time.time_ns()
        prefill = batch.prefilling
        if prefill:
            batch.prepare_for_prefill()
        else:
            batch.prepare_for_decode(self.dtype, self.use_contiguous_pa)
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
            # pad in left
            if batch.prefill_cache_indices is not None:
                batch.position_ids = batch.position_ids[batch.prefill_cache_indices][
                    indices
                ]
            else:
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
        # one, we need to first do a HPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        # Cumulative length
        cu_accepted_ids = accepted_ids.new_zeros(accepted_ids.shape[0] + 1)
        torch.cumsum(accepted_ids, dim=0, out=cu_accepted_ids[1:])
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
            if not request_is_prefilling:
                # Only save tokens if we are done prefilling for this request
                batch.all_input_ids_tensor[
                    i,
                    batch.cache_lengths_tensor[i]
                    + batch.input_lengths[i] : batch.cache_lengths_tensor[i]
                    + batch.input_lengths[i]
                    + accepted_ids[i],
                ] = next_input_ids[cu_accepted_ids[i] : cu_accepted_ids[i + 1]]
            cumulative_length += input_length

        # Update values
        # These values can be updated without a HPU -> CPU sync
        if not prefill or (prefill and finished_prefilling):
            batch.input_ids = next_input_ids[cu_accepted_ids[1:] - 1]
            batch.speculative_ids = speculative_ids
            if batch.position_ids.dim() == 2:
                # Qwen2_vl case:
                batch.position_ids += accepted_ids.unsqueeze(-1)
            else:
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
            # HPU <-> CPU sync
            prefill_logprobs = prefill_logprobs.view(-1).tolist()

        # Does a HPU <-> CPU sync internally
        if prefill and finished_prefilling:
            # adjust segment lengths to account for all request lengths being 1 during decoding
            adapter_segments, _ = find_segments(batch.adapter_meta.adapter_indices)
            batch.adapter_meta.adapter_segments = torch.tensor(
                adapter_segments,
                dtype=torch.int32,
                device=batch.adapter_meta.adapter_segments.device,
            )

        # HPU <-> CPU sync
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
