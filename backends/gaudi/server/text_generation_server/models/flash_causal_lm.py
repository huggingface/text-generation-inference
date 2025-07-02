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
    pad_next_token_chooser_parameters,
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
    KVCompressCache,
    Seqlen,
    HPUPagedAttentionMetadata,
    trim_attn_metadata,
    trim_seqlen_metadata,
    _async_h2d_tensor_copy,
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
from text_generation_server.utils.prefill_chunking import (
    get_max_prefill_tokens,
)
import vllm_hpu_extension.environment as environment
import habana_frameworks.torch as htorch
import itertools
from vllm_hpu_extension.bucketing.common import get_bucketing_context
from vllm_hpu_extension.profiler import HabanaMemoryProfiler, format_bytes

tracer = trace.get_tracer(__name__)


def generate_block_metadata(
    dtype,
    use_contiguous_pa,
    slots,
    block_tables,
    bucketing_ctx,
    slots_in_window=None,
    block_bucket_size=None,
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

    last_block_usage = [slot % BLOCK_SIZE + 1 for slot in slots]
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
        if block_bucket_size is None:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            if bucketing_ctx is not None:
                block_bucket_size = bucketing_ctx.get_padded_decode_num_blocks(
                    block_bucket_size
                )
        indices: List[Any]
        indices = [None] * block_bucket_size
        for i, bid in enumerate(block_list):
            indices[bid] = i
        block_list = gather_list(block_list, indices, 0)
        block_groups = gather_list(block_groups, indices, -1)
        block_usage = gather_list(block_usage, indices, 1)
    else:
        if block_bucket_size is None:
            block_bucket_size = len(block_list)
            if bucketing_ctx is not None:
                block_bucket_size = bucketing_ctx.get_padded_decode_num_blocks(
                    block_bucket_size
                )
        block_list = pad_list(block_list, block_bucket_size, 0)
        block_groups = pad_list(block_groups, block_bucket_size, -1)
        block_usage = pad_list(block_usage, block_bucket_size, 1)
    slots_in_window_mask = None
    if slots_in_window is not None:
        slot_list = [
            block_id * BLOCK_SIZE + slot_idx
            for block_id in block_list
            for slot_idx in range(BLOCK_SIZE)
        ]
        slot_list = torch.tensor(slot_list, dtype=torch.int64)
        slot_list = slot_list.view(-1, BLOCK_SIZE)
        slots_in_window_mask = torch.isin(slot_list, slots_in_window)
        for i in range(slots_in_window_mask.shape[0]):
            if not slots_in_window_mask[i].any():
                slots_in_window_mask[i, 0] = True

    block_list = torch.tensor(block_list, dtype=torch.int, device="cpu")
    block_groups = torch.tensor(block_groups, dtype=torch.int, device="cpu")
    block_usage = torch.tensor(block_usage, dtype=dtype, device="cpu")
    return (
        block_list,
        block_groups,
        block_usage,
        slots_in_window_mask,
        block_bucket_size,
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

    next_token_logits: Optional[torch.Tensor]
    speculative_logits: Optional[torch.Tensor]
    valid_indices: Optional[List[int]]

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
            else:
                assert False, "prefill_logprobs not supported yet"
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

        # put on cpu temporarily, move to hpu in prepare_for_prefill
        all_input_ids_tensor = torch.tensor(all_input_ids_tensor, dtype=torch.int64)

        top_n_tokens_tensor = torch.tensor(top_n_tokens, dtype=torch.int64)

        block_tables_ragged = torch.tensor(block_tables_ragged, dtype=torch.int32)
        cu_blocks = torch.tensor(cu_blocks, dtype=torch.int64)
        block_tables_tensor = torch.empty(
            (len(block_tables), max_blocks),
            dtype=torch.int32,
        )

        for i, request_blocks in enumerate(block_tables):
            block_tables_tensor[i, : len(request_blocks)] = torch.tensor(request_blocks)

        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.int32)

        slots = torch.tensor(slots, dtype=torch.int64)
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
            next_token_logits=None,
            speculative_logits=None,
            valid_indices=None,
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
        slot_filtering_indices = torch.zeros(self.slots.shape[0], dtype=torch.bool)

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

        block_tables_tensor = self.block_tables_tensor[indices]
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
            input_lengths_tensor = self.input_lengths_tensor[indices]
            cache_lengths_tensor = self.cache_lengths_tensor[indices]

            # Move to GPU now that we have the whole tensor
            slot_indices = slot_indices.to(device)
            if self.adapter_meta is not None:
                adapter_indices = self.adapter_meta.adapter_indices[indices]
                adapter_segments, adapter_segment_indices = find_segments(
                    adapter_indices
                )
                adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32)
                adapter_meta = AdapterBatchMetadata(
                    adapter_indices=adapter_indices,
                    adapter_set=adapter_set,
                    adapter_segments=adapter_segments,
                    segment_indices=adapter_segment_indices,
                )
            else:
                adapter_meta = None
        htorch.core.mark_step()
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
            all_input_ids_tensor=self.all_input_ids_tensor,
            next_token_chooser=self.next_token_chooser,
            stopping_criterias=stopping_criterias,
            top_n_tokens=self.top_n_tokens,
            top_n_tokens_tensor=self.top_n_tokens_tensor,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            speculative_ids=self.speculative_ids,
            adapter_meta=adapter_meta,
            hpu_attn_meta=None,
            valid_indices=indices,
            next_token_logits=self.next_token_logits,
            speculative_logits=self.speculative_logits,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(
        cls, batches: List["FlashCausalLMBatch"], padded_total_bs: int = 0
    ) -> "FlashCausalLMBatch":
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
        ADAPTER_TO_INDEX = get_adapter_to_index()
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
            if padded_total_bs == batches[0].input_ids.shape[0]:
                input_ids = batches[0].input_ids
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
            if ADAPTER_TO_INDEX:
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
        all_input_ids_tensor = batches[0].all_input_ids_tensor
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
            valid_bsize = len(batch)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + cumulative_batch_size

            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + valid_bsize

            index = torch.tensor(list(range(start_index, end_index)), device="cpu")
            top_n_tokens_tensor.index_copy_(0, index, batch.top_n_tokens_tensor)
            if i > 0:
                all_input_ids_tensor.index_copy_(
                    0,
                    index.to(batch.all_input_ids_tensor.device),
                    batch.all_input_ids_tensor[:valid_bsize, :],
                )

            block_tables_tensor[
                start_index:end_index, : batch.block_tables_tensor.shape[1]
            ] = batch.block_tables_tensor[:, :max_blocks]
            prompt_lengths_tensor.index_copy_(0, index, batch.prompt_lengths_tensor)

            slots_start_index = cumulative_slots
            slots_end_index = cumulative_slots + len(batch.slots)
            slot_index = torch.tensor(
                list(range(slots_start_index, slots_end_index)),
                device=batch.slots.device,
            )

            slots.index_copy_(0, slot_index, batch.slots)
            cu_slots[start_index + 1 : end_index + 1] = (
                batch.cu_slots[1:] + cumulative_slots
            )

            if not prefilling:
                if padded_total_bs != batches[0].input_ids.shape[0] or i > 0:
                    input_ids.index_copy_(
                        0, index.to(input_ids.device), batch.input_ids[:valid_bsize]
                    )
                position_ids.index_copy_(0, index, batch.position_ids[:valid_bsize])
                slot_indices.index_copy_(
                    0, index, batch.slot_indices + cumulative_slots
                )
                input_lengths_tensor.index_copy_(
                    0, index, batch.input_lengths_tensor[:valid_bsize]
                )
                cache_lengths_tensor.index_copy_(
                    0, index, batch.cache_lengths_tensor[:valid_bsize]
                )
                if ADAPTER_TO_INDEX:
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

        if ADAPTER_TO_INDEX and adapter_segment_builder is not None:
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
            adapter_meta=adapter_meta if ADAPTER_TO_INDEX else None,
            hpu_attn_meta=None,
            next_token_logits=None,
            speculative_logits=None,
            valid_indices=None,
        )

    def prepare_for_decode(
        self, dtype, use_contiguous_pa, bucketing_ctx, pad_token_id, sliding_window
    ):
        block_num = [length // BLOCK_SIZE + 1 for length in self.cache_lengths]
        block_tables = []
        for i, bt in enumerate(self.block_tables):
            block_tables.append(bt[0 : block_num[i]])
        if bucketing_ctx is not None:
            padded_bs = bucketing_ctx.get_padded_decode_batch_size(
                self.input_ids.shape[0]
            )
        else:
            padded_bs = self.input_ids.shape[0]
        slots = self.slots[self.slot_indices]

        block_list, block_groups, block_usage, _, block_bucket_size = (
            generate_block_metadata(
                dtype,
                use_contiguous_pa,
                slots,
                block_tables,
                bucketing_ctx,
            )
        )
        meta = HPUPagedAttentionMetadata(
            block_list=_async_h2d_tensor_copy(block_list),
            block_groups=_async_h2d_tensor_copy(block_groups),
            block_usage=_async_h2d_tensor_copy(block_usage),
            block_mapping=None,
            attn_bias=None,
        )
        if sliding_window is not None:
            block_tables_in_window = []
            for i, bt in enumerate(self.block_tables):
                block_num_in_window = (
                    sliding_window + 2 * BLOCK_SIZE - 2 - slots[i] % BLOCK_SIZE
                ) // BLOCK_SIZE
                block_tables_in_window.append(
                    bt[max(0, block_num[i] - block_num_in_window) : block_num[i]]
                )
            slots_in_window = []
            for i, indice in enumerate(self.slot_indices):
                start_idx = indice - self.cache_lengths[i]
                mask = (
                    indice
                    - torch.arange(
                        start_idx,
                        indice + 1,
                        device=self.slots.device,
                    )
                ) < sliding_window
                slots_in_window.append(self.slots[start_idx : indice + 1][mask])
            slots_in_window = torch.cat(slots_in_window, dim=0)
            (
                block_list_in_window,
                block_groups_in_window,
                block_usage_in_window,
                slots_in_window_mask,
                _,
            ) = generate_block_metadata(
                dtype,
                use_contiguous_pa,
                slots,
                block_tables_in_window,
                bucketing_ctx,
                slots_in_window,
                block_bucket_size,
            )
            meta.block_list_in_window = _async_h2d_tensor_copy(block_list_in_window)
            meta.block_groups_in_window = _async_h2d_tensor_copy(block_groups_in_window)
            meta.block_usage_in_window = _async_h2d_tensor_copy(block_usage_in_window)
            meta.slots_in_window_mask = _async_h2d_tensor_copy(slots_in_window_mask)

        self.hpu_attn_meta = trim_attn_metadata(meta)
        self.input_ids = F.pad(
            self.input_ids, (0, padded_bs - self.input_ids.shape[0]), value=pad_token_id
        )

        if self.position_ids.dim() == 2:
            # Qwen VL case
            self.position_ids = F.pad(
                self.position_ids,
                (0, 0, 0, padded_bs - self.position_ids.shape[0]),
                value=1,
            )
        else:
            self.position_ids = F.pad(
                self.position_ids, (0, padded_bs - self.position_ids.shape[0]), value=1
            )
        self.input_lengths_tensor = F.pad(
            self.input_lengths_tensor,
            (0, padded_bs - self.input_lengths_tensor.shape[0]),
            value=0,
        )
        self.cache_lengths_tensor = F.pad(
            self.cache_lengths_tensor,
            (0, padded_bs - self.cache_lengths_tensor.shape[0]),
            value=0,
        )
        if len(self.next_token_chooser.do_sample) != padded_bs:
            next_token_chooser_parameters = []
            next_token_chooser_parameters.extend([r.parameters for r in self.requests])
            pad_next_token_chooser_parameters(next_token_chooser_parameters, padded_bs)
            # update past grammar states
            fsm_grammar_states = [0] * padded_bs

            for i, req in enumerate(self.requests):
                fsm_grammar_states[i] = self.next_token_chooser.fsm_grammar_states[i]

            self.next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
                next_token_chooser_parameters,
                self.next_token_chooser.dtype,
                self.next_token_chooser.device,
                self.next_token_chooser.tokenizer,
                fsm_grammar_states,
            )

    def prepare_for_prefill(
        self, max_padded_input_len, max_padded_bs, max_total_tokens, pad_token_id
    ):
        # Prepare values if we need to continue prefilling
        # Speculation must be ignored while we prefill even with chunking
        # it simplifies everything
        assert self.speculative_ids is None

        # device = self.block_tables_tensor.device

        # hpu does not support varlen for prefill, use sdpa instead. so need to pad input_tensor, position
        # padding to left to work with sliding window
        # use prefill_cache_indices to indicate the valid kv slot, update prefill_next_token_indices to indicate
        # the right logit position
        input_ids_padded_length = []
        # need extra pad to match warmup seq
        extra_pad = max_padded_input_len - self.max_input_length
        extra_pad_bs = max_padded_bs - len(self)
        device = "hpu"
        if isinstance(self.input_ids, list) and len(self) > 1:
            input_ids_padded_length = []
            input_ids = []
            for input_id in self.input_ids:
                padded = self.max_input_length - len(input_id) + extra_pad
                if padded > 0:
                    input_id = [pad_token_id] * padded + input_id
                input_ids.append(input_id)
                input_ids_padded_length.append(padded)
            input_ids = np.concatenate(input_ids, dtype=np.int64)
            self.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        elif isinstance(self.input_ids, list):
            input_ids = self.input_ids[0]
            input_ids_padded_length.append(extra_pad)
            input_ids = [pad_token_id] * extra_pad + input_ids
            self.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        else:
            input_ids = torch.full(
                (max_padded_input_len * len(self),),
                pad_token_id,
                dtype=torch.int64,
                device=self.input_ids.device,
            )
            src_pos = 0
            for i in range(len(self)):
                end_pos = (i + 1) * max_padded_input_len
                start_pos = end_pos - self.input_lengths[i]
                input_ids[start_pos:end_pos] = self.input_ids[
                    src_pos : src_pos + self.input_lengths[i]
                ]
                input_ids_padded_length.append(
                    max_padded_input_len - self.input_lengths[i]
                )
                src_pos += self.input_lengths[i]
            self.input_ids = input_ids

        self.input_ids = F.pad(
            self.input_ids, (0, extra_pad_bs * max_padded_input_len), value=pad_token_id
        )

        self.input_lengths_tensor = torch.tensor(self.input_lengths, dtype=torch.int32)

        self.input_lengths_tensor = F.pad(
            self.input_lengths_tensor, (0, extra_pad_bs), value=0
        )

        cu_seqlen_prefill = self.input_lengths_tensor.new_zeros(max_padded_bs + 1)
        torch.cumsum(self.input_lengths_tensor, out=cu_seqlen_prefill[1:], dim=0)
        self.cu_seqlen_prefill = cu_seqlen_prefill.to(torch.int32)
        self.cache_lengths_tensor = torch.tensor(self.cache_lengths, dtype=torch.int32)
        self.cache_lengths_tensor = F.pad(
            self.cache_lengths_tensor, (0, extra_pad_bs), value=0
        )

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
                            dtype=torch.int32,
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
                            dtype=torch.int32,
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

        self.position_ids = position_ids
        self.position_ids = F.pad(
            self.position_ids, (0, extra_pad_bs * max_padded_input_len), value=1
        )
        self.slot_indices = slot_indices

        self.prefill_cu_outlens = prefill_cu_outlens
        self.prefill_cache_indices = torch.zeros_like(
            self.input_ids, dtype=torch.bool, device="cpu"
        )
        self.prefill_cache_indices[prefill_cache_indices] = True

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = self.cu_seqlen_prefill[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = self.cu_seqlen_prefill[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.cat(prefill_head_indices)
            prefill_next_token_indices = torch.tensor(
                prefill_next_token_indices, dtype=torch.int64
            )

        self.prefill_head_indices = prefill_head_indices
        self.prefill_next_token_indices = prefill_next_token_indices
        input_ids_padded_length_tensor = torch.cumsum(
            torch.tensor(input_ids_padded_length, dtype=torch.int32),
            dim=-1,
        ).to(torch.int32)
        input_ids_padded_length_tensor = F.pad(
            input_ids_padded_length_tensor, (0, extra_pad_bs), value=0
        )
        if self.prefill_head_indices is not None:
            self.prefill_head_indices = (
                self.prefill_head_indices + input_ids_padded_length_tensor
            )

        if self.prefill_next_token_indices is not None:
            self.prefill_next_token_indices = (
                self.prefill_next_token_indices + input_ids_padded_length_tensor
            )
        all_input_ids_tensor = torch.full(
            (max_padded_bs, max(max_total_tokens, self.all_input_ids_tensor.shape[-1])),
            pad_token_id,
            dtype=torch.int64,
            device="hpu",
        )
        for i in range(len(self)):
            all_input_ids_tensor[i, : self.all_input_ids_tensor.shape[-1]] = (
                self.all_input_ids_tensor[i]
            )
        self.all_input_ids_tensor = all_input_ids_tensor
        if len(self.next_token_chooser.do_sample) != max_padded_bs:
            next_token_chooser_parameters = []
            next_token_chooser_parameters.extend([r.parameters for r in self.requests])
            pad_next_token_chooser_parameters(
                next_token_chooser_parameters, max_padded_bs
            )
            # update past grammar states
            fsm_grammar_states = [0] * max_padded_bs

            for i, req in enumerate(self.requests):
                fsm_grammar_states[i] = self.next_token_chooser.fsm_grammar_states[i]

            self.next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
                next_token_chooser_parameters,
                self.next_token_chooser.dtype,
                self.next_token_chooser.device,
                self.next_token_chooser.tokenizer,
                fsm_grammar_states,
            )

        if ADAPTER_TO_INDEX:
            if adapter_set:
                adapter_indices = torch.cat(adapter_indices_list).to(dtype=torch.int64)
                adapter_segments, adapter_segment_indices = find_segments(
                    adapter_indices
                )
            else:
                adapter_indices = torch.zeros_like(self.input_ids)
                adapter_segments = [0, len(adapter_indices)]
                adapter_segment_indices = [len(adapter_indices) - 1]

            adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32)
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
        if world_size > 1:
            self.process_group_cpu = torch.distributed.new_group(backend="gloo")

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

        if getattr(config, "sliding_window", None) is None:
            config.sliding_window = None
        if getattr(config, "use_sliding_window", True) is False:
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
            if num_kv_heads // self.process_group.size() > 0
            else num_kv_heads
        )
        assert self.num_kv_heads > 0

        if head_size is None:
            # Some models use GQA and different sizes for o_proj
            # and q_proj, that allows for that.
            if getattr(config, "head_dim", None) is not None:
                self.head_size = config.head_dim
            else:
                self.head_size = config.hidden_size // config.num_attention_heads
        else:
            self.head_size = head_size

        self.cuda_graphs = {}
        self.kv_cache = []
        self.kv_cache_dtype = dtype if kv_cache_dtype is None else kv_cache_dtype
        self.bucketing_ctx = None
        self.max_total_tokens = None
        self.max_input_tokens = None
        htorch.core.hpu_set_env()
        if htorch.utils.internal.is_lazy():
            htorch.hpu.wrap_in_hpu_graph(model, disable_tensor_cache=True)
        environment.set_model_config(self.config)
        self.use_contiguous_pa = (
            os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() == "true"
        )
        self.limit_hpu_graph = (
            os.environ.get("LIMIT_HPU_GRAPH", "false").lower() == "true"
        )
        self.skip_warmup = os.getenv("VLLM_SKIP_WARMUP", "false").lower() == "true"
        self.max_seq_len_to_capture = 8192
        if tokenizer.pad_token_id is None:
            if config.pad_token_id is not None:
                tokenizer.pad_token_id = config.pad_token_id
            elif config.eos_token_id is not None:
                tokenizer.pad_token_id = (
                    config.eos_token_id[0]
                    if isinstance(config.eos_token_id, list)
                    else config.eos_token_id
                )
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0
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
        if self.config.model_type == "deepseek_v3":
            self.kv_cache = [
                KVCompressCache(
                    num_blocks=num_blocks,
                    head_size=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        else:
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
        if os.environ.get("MAX_BATCH_SIZE") is None:
            raise RuntimeError(
                "MAX_BATCH_SIZE is not set, it should be set in the launcher "
                "using `--max-batch-size xxx`"
            )
        # The warmup batch is the biggest batch we could ever receive
        self.kv_cache = []
        empty_cache()
        self.graphed_buckets = set()
        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.kv_cache_dtype).element_size()
        if self.config.model_type == "deepseek_v3":
            cache_block_size = BLOCK_SIZE * (
                self.config.kv_lora_rank + self.config.qk_rope_head_dim
            )
        else:
            cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
            cache_block_size = cache_block_size * 2
        total_cache_size = self.num_layers * cache_block_size * dtype_size
        free_memory = get_free_memory(self.device, TGI_WIGGLE_ROOM)
        self.mem_reserved = int(free_memory * (1 - MEMORY_FRACTION))
        graph_reserved_mem = (
            float(os.environ.get("TGI_GRAPH_RESERVED_MEM", "0.1"))
            if htorch.utils.internal.is_lazy()
            else 0
        )
        mem_used_from_graph = int(
            (free_memory - self.mem_reserved) * graph_reserved_mem
        )
        log_master(
            logger.info,
            f"Free memory on device {self.device}: {format_bytes(free_memory)} used_for_graph: {format_bytes(mem_used_from_graph)} ratio {graph_reserved_mem} reserved_for_runtime: {format_bytes(self.mem_reserved)}",
        )
        if max_total_tokens is None:
            max_total_tokens = sum(batch.input_lengths)

        if max_input_tokens is None:
            max_input_tokens = max_total_tokens - 1

        self.max_total_tokens = max_total_tokens
        self.max_input_tokens = max_input_tokens
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
            _, _batch, _ = self.generate_token([batch])
        except Exception:
            raise RuntimeError(
                f"Not enough memory to handle {num_tokens} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            )

        synchronize(self.device)
        free_memory = get_free_memory(self.device, TGI_WIGGLE_ROOM)

        kv_memory = free_memory - self.mem_reserved - mem_used_from_graph
        num_blocks = (
            # Leave 5% for some wiggle room
            int(kv_memory // total_cache_size)
            # Add batch.num_blocks as we allocated it above, so it is included in the peak memory.
            + batch_num_blocks
        )

        log_master(logger.info, f"KV-cache blocks: {num_blocks}, size: {BLOCK_SIZE}")

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
        self.max_batch_prefill_tokens = get_max_prefill_tokens()
        max_num_seqs = int(os.getenv("MAX_BATCH_SIZE"))
        HPUBucketingContext = get_bucketing_context()
        # need to warmup one more step since block is allocated from 1
        block_step = os.getenv("VLLM_DECODE_BLOCK_BUCKET_STEP", BLOCK_SIZE)
        max_total_tokens_aligned = math.ceil(
            max_total_tokens / BLOCK_SIZE
        ) * BLOCK_SIZE + math.ceil(block_step * BLOCK_SIZE / max_num_seqs)
        model_max_length = self.tokenizer.model_max_length
        max_position_embeddings = getattr(
            self.config, "max_position_embeddings", model_max_length
        )

        self.bucketing_ctx = HPUBucketingContext(
            max_num_seqs,
            max_num_seqs,  # self.max_num_prefill_seqs, #TODO
            BLOCK_SIZE,
            max_num_seqs * max_total_tokens_aligned,
            False,
            min(model_max_length, max_position_embeddings),
            max_input_tokens,
            max_total_tokens_aligned,
        )
        max_blocks = max(
            BLOCK_SIZE, max_num_seqs * max_total_tokens_aligned // BLOCK_SIZE
        )
        self.bucketing_ctx.num_hpu_blocks = min(max_blocks, num_blocks)
        synchronize(self.device)
        if self.skip_warmup:
            self.bucketing_ctx.generate_prompt_buckets()
            self.bucketing_ctx.generate_decode_buckets(
                self.bucketing_ctx.num_hpu_blocks
            )
            log_master(
                logger.info, "skip warmup hpu graph, not recommmended, may cause OOM"
            )
            del _batch, batch
            return int(num_blocks * BLOCK_SIZE), max_input_tokens, max_total_tokens
        self.warmup_hpu_graph(batch)
        del _batch, batch

        return int(num_blocks * BLOCK_SIZE), max_input_tokens, max_total_tokens

    def log_warmup(self, prefilling, i, max_i, batch_size, seq_len):
        free_mem = format_bytes(HabanaMemoryProfiler.current_free_device_memory())
        phase = "Prompt" if prefilling else "Decode"
        dim = "seq_len" if prefilling else "num_blocks"
        graphed_bucket = (batch_size, seq_len, prefilling)
        bypass = graphed_bucket not in self.graphed_buckets
        msg = (
            f"[Warmup][{phase}][{i+1}/{max_i}] "
            f"batch_size:{batch_size} "
            f"{dim}:{seq_len} "
            f"bypass:{bypass} "
            f"free_mem:{free_mem}"
            ", this may take a while..."
        )
        log_master(logger.info, msg)

    def use_graphs(self, prefill, seq_len, batch_size):
        if self.limit_hpu_graph and prefill:
            return False

        if self.skip_warmup:
            return True

        return (batch_size, seq_len, prefill) in self.graphed_buckets

    def align_workers(self, value, op):
        if self.world_size <= 1:
            return value
        value_t = torch.tensor(value, device="cpu")
        torch.distributed.all_reduce(value_t, op=op, group=self.process_group_cpu)
        return value_t.item()

    def warmup_hpu_graph(self, batch):
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

    def warmup_prefill(
        self, prompt_len: int, batch_size: int, batch: FlashCausalLMBatch
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

        input_lengths = torch.ones(batch_size, dtype=torch.int32) * prompt_len
        cu_seqlen_prefill = torch.zeros(batch_size + 1, dtype=torch.int32)
        torch.cumsum(input_lengths, -1, out=cu_seqlen_prefill[1:])

        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )
        lm_head_indices = input_lengths - 1
        kwargs = {}
        if htorch.utils.internal.is_lazy():
            kwargs["bypass_hpu_graphs"] = not self.use_graphs(
                True, prompt_len, batch_size
            )
        if self.sliding_window is not None:
            attn_mask = seqlen.make_sliding_window_bias(
                input_lengths.tolist(),
                self.sliding_window,
                self.dtype,
                prompt_len,
                batch_size,
            )
            seqlen.attn_mask = _async_h2d_tensor_copy(attn_mask)

        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            input_ids=_async_h2d_tensor_copy(input_ids),
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=_async_h2d_tensor_copy(cu_seqlen_prefill),
            kv_cache=self.kv_cache,
            slots=_async_h2d_tensor_copy(slots),
            seqlen=trim_seqlen_metadata(seqlen),
            lm_head_indices=_async_h2d_tensor_copy(lm_head_indices),
            adapter_data=None,
            hpu_attention_meta=None,
            **kwargs,
        )

    def warmup_decode(self, batch_size: int, block_num: int, batch: FlashCausalLMBatch):
        input_ids = torch.zeros(batch_size, dtype=batch.input_ids.dtype)
        position_ids = torch.arange(batch_size, dtype=batch.position_ids.dtype)
        blocks = [block_num // batch_size for _ in range(batch_size)]
        blocks[0] += block_num % batch_size
        block_tables = []
        slots = []
        start_idx = 0
        slot_indices = []

        # fetch the last blocked to warmup block num
        for i in range(batch_size):
            block_array = list(range(start_idx, start_idx + blocks[i]))
            slots.append(BLOCK_SIZE * block_array[-1] + BLOCK_SIZE - 1)
            slot_indices.append((start_idx + blocks[i]) * BLOCK_SIZE - 1)
            block_tables.append(block_array)
            start_idx += blocks[i]
        input_lengths = torch.ones(batch_size, dtype=torch.int32)
        cu_seqlen_prefill = torch.zeros(batch_size + 1, dtype=torch.int32)
        torch.cumsum(input_lengths, -1, out=cu_seqlen_prefill[1:])

        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )
        block_list, block_groups, block_usage, _, block_bucket_size = (
            generate_block_metadata(
                self.dtype,
                self.use_contiguous_pa,
                slots,
                block_tables,
                self.bucketing_ctx,
            )
        )
        meta = HPUPagedAttentionMetadata(
            block_list=_async_h2d_tensor_copy(block_list),
            block_groups=_async_h2d_tensor_copy(block_groups),
            block_usage=_async_h2d_tensor_copy(block_usage),
            block_mapping=None,
            attn_bias=None,
        )
        if self.sliding_window is not None:
            block_tables_in_window = []
            for i, bt in enumerate(block_tables):
                block_num_in_window = (
                    self.sliding_window + BLOCK_SIZE - 1
                ) // BLOCK_SIZE
                block_tables_in_window.append(
                    bt[max(0, blocks[i] - block_num_in_window) : blocks[i]]
                )
            slots_in_window = []
            start_idx = 0
            for i, indice in enumerate(slot_indices):
                mask = (
                    indice - torch.arange(start_idx, indice + 1)
                ) < self.sliding_window
                slots_in_window.append(torch.arange(start_idx, indice + 1)[mask])
                start_idx += blocks[i] * BLOCK_SIZE
            slots_in_window = torch.cat(slots_in_window, dim=0)
            (
                block_list_in_window,
                block_groups_in_window,
                block_usage_in_window,
                slots_in_window_mask,
                _,
            ) = generate_block_metadata(
                self.dtype,
                self.use_contiguous_pa,
                slots,
                block_tables_in_window,
                self.bucketing_ctx,
                slots_in_window,
                block_bucket_size,
            )
            meta.block_list_in_window = _async_h2d_tensor_copy(block_list_in_window)
            meta.block_groups_in_window = _async_h2d_tensor_copy(block_groups_in_window)
            meta.block_usage_in_window = _async_h2d_tensor_copy(block_usage_in_window)
            meta.slots_in_window_mask = _async_h2d_tensor_copy(slots_in_window_mask)

        hpu_attention_meta = trim_attn_metadata(meta)
        slots_tensor = torch.tensor(slots, dtype=batch.slots.dtype)
        kwargs = {}
        if htorch.utils.internal.is_lazy():
            kwargs["bypass_hpu_graphs"] = not self.use_graphs(
                False, hpu_attention_meta.block_list.shape[0], batch_size
            )
        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            input_ids=_async_h2d_tensor_copy(input_ids),
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=None,
            kv_cache=self.kv_cache,
            slots=_async_h2d_tensor_copy(slots_tensor),
            seqlen=trim_seqlen_metadata(seqlen),
            lm_head_indices=None,
            adapter_data=None,
            hpu_attention_meta=hpu_attention_meta,
            **kwargs,
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
        if batch.prefill_cache_indices is not None:
            slots_pad = torch.zeros_like(input_ids, device=slots.device)
            slots_pad[batch.prefill_cache_indices] = slots
            slots = slots_pad
        else:
            slots_pad = torch.zeros_like(input_ids, device=slots.device)
            slots_pad[: slots.shape[0]] = slots
            slots = slots_pad
        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )

        kwargs = {}
        batch_size = input_lengths.shape[0]
        prompt_len = (
            input_ids.shape[0] // batch_size
            if batch.prefilling
            else batch.hpu_attn_meta.block_list.shape[0]
        )
        if htorch.utils.internal.is_lazy():
            kwargs["bypass_hpu_graphs"] = not self.use_graphs(
                batch.prefilling, prompt_len, batch_size
            )
        if self.sliding_window is not None and batch.prefilling:
            attn_mask = seqlen.make_sliding_window_bias(
                input_lengths.tolist(),
                self.sliding_window,
                self.dtype,
                prompt_len,
                batch_size,
            )
            seqlen.attn_mask = _async_h2d_tensor_copy(attn_mask)

        logits, speculative_logits = self.model.forward(
            input_ids=input_ids,
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=_async_h2d_tensor_copy(cu_seqlen_prefill),
            kv_cache=kv_cache,
            slots=_async_h2d_tensor_copy(slots),
            seqlen=trim_seqlen_metadata(seqlen),
            lm_head_indices=_async_h2d_tensor_copy(lm_head_indices),
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

        # In order to pipeline any actions on CPU we perform the operation in 3 main stages:
        # Stage 1. Collect next token ids of any previously started generations
        start = time.time_ns()
        prev_batches = []
        requests_to_generate = []
        for batch_id, batch in enumerate(batches):
            if batch.next_token_logits is not None:
                prefill = batch.prefilling
                if batch.prefilling:
                    batch.prefilling = False
                    batch.prefilling_mask = [False] * len(batch)

                speculate = get_speculate()
                (
                    next_input_ids,
                    next_token_logprobs,
                    logprobs,
                    accepted_ids,
                    speculative_ids,
                ) = batch.next_token_chooser(
                    batch.all_input_ids_tensor[
                        : batch.next_token_logits.shape[0], : batch.max_current_length
                    ],
                    batch.next_token_logits,
                    speculate,
                    batch.speculative_ids,
                    batch.speculative_logits,
                )

                batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
                    batch.top_n_tokens,
                    _async_h2d_tensor_copy(batch.top_n_tokens_tensor),
                    logprobs,
                    accepted_ids,
                )
                if batch.valid_indices is not None:
                    # TODO speculative decoding handling missing
                    index = torch.arange(
                        0,
                        len(batch.valid_indices),
                        device=batch.all_input_ids_tensor.device,
                    )
                    batch.all_input_ids_tensor.index_copy_(
                        0, index, batch.all_input_ids_tensor[batch.valid_indices]
                    )
                    padded_total_bs = self.bucketing_ctx.get_padded_decode_batch_size(
                        len(batch.valid_indices)
                    )
                    next_input_ids.index_copy_(
                        0, index, next_input_ids[batch.valid_indices]
                    )
                    next_input_ids = next_input_ids[:padded_total_bs]

                    next_token_logprobs.index_copy_(
                        0, index, next_token_logprobs[batch.valid_indices]
                    )
                    accepted_ids.index_copy_(
                        0, index, accepted_ids[batch.valid_indices]
                    )
                    if speculative_ids is not None:
                        speculative_ids = speculative_ids[batch.valid_indices]
                    batch.top_n_tokens_tensor = batch.top_n_tokens_tensor[
                        batch.valid_indices
                    ]
                    top_n_tokens = []
                    batch_top_token_ids_v = []
                    batch_top_token_logprobs_v = []
                    for i in batch.valid_indices:
                        top_n_tokens.append(batch.top_n_tokens[i])
                        batch_top_token_ids_v.append(batch_top_token_ids[i])
                        batch_top_token_logprobs_v.append(batch_top_token_logprobs[i])
                    batch_top_token_ids = batch_top_token_ids_v
                    batch_top_token_logprobs = batch_top_token_logprobs_v
                    batch.top_n_tokens = top_n_tokens
                    batch.next_token_chooser = batch.next_token_chooser.filter(
                        batch.valid_indices
                    )
                    batch.valid_indices = None

                # Since we are done prefilling, all the tensors that were concatenating values for all the requests
                # instantly become of shape [BATCH_SIZE]
                if prefill:
                    indices = batch.cu_seqlen_prefill[1:] - 1
                    # pad in left
                    if batch.prefill_cache_indices is not None:
                        batch.position_ids = batch.position_ids[
                            batch.prefill_cache_indices
                        ][indices]
                    else:
                        batch.position_ids = batch.position_ids[indices]

                    batch.slot_indices = batch.slot_indices[indices[: len(batch)]]
                    if batch.adapter_meta is not None:
                        batch.adapter_meta.adapter_indices = (
                            batch.adapter_meta.adapter_indices[indices]
                        )
                # For each member of the batch
                # Cumulative length

                if batch.speculative_logits is not None:
                    cu_accepted_ids = accepted_ids.new_zeros(accepted_ids.shape[0] + 1)
                    torch.cumsum(accepted_ids, dim=0, out=cu_accepted_ids[1:])
                    for i in range(len(batch)):
                        batch.all_input_ids_tensor[
                            i,
                            batch.cache_lengths[i]
                            + batch.input_lengths[i] : batch.cache_lengths[i]
                            + batch.input_lengths[i]
                            + accepted_ids[i],
                        ] = next_input_ids[cu_accepted_ids[i] : cu_accepted_ids[i + 1]]
                    batch.input_ids = next_input_ids[cu_accepted_ids[1:] - 1]
                    accepted_ids = accepted_ids.cpu()
                    if batch.position_ids.dim() == 2:
                        # Qwen2_vl case:
                        batch.position_ids += accepted_ids.unsqueeze(-1)
                    else:
                        batch.position_ids += accepted_ids
                    batch.cache_lengths_tensor += (
                        batch.input_lengths_tensor + accepted_ids - 1
                    )
                    batch.input_lengths_tensor = torch.ones_like(
                        batch.input_lengths_tensor
                    )
                    batch.slot_indices += accepted_ids[: len(batch)]
                else:
                    index = batch.cache_lengths_tensor + batch.input_lengths_tensor
                    index = F.pad(
                        index, (0, next_input_ids.shape[0] - index.shape[0]), value=0
                    )
                    index = index.to(batch.all_input_ids_tensor.device)
                    batch_idx = torch.arange(
                        0,
                        index.shape[0],
                        dtype=torch.long,
                        device=batch.all_input_ids_tensor.device,
                    )
                    batch.all_input_ids_tensor.index_put_(
                        (batch_idx, index.long()), next_input_ids
                    )
                    batch.input_ids = next_input_ids
                    batch.position_ids += 1
                    batch.cache_lengths_tensor += batch.input_lengths_tensor
                    batch.input_lengths_tensor = torch.ones_like(
                        batch.input_lengths_tensor
                    )
                    batch.slot_indices += 1

                batch.speculative_ids = speculative_ids

                # Does a HPU <-> CPU sync internally
                if prefill and batch.adapter_meta is not None:
                    # adjust segment lengths to account for all request lengths being 1 during decoding
                    adapter_segments, _ = find_segments(
                        batch.adapter_meta.adapter_indices
                    )
                    batch.adapter_meta.adapter_segments = torch.tensor(
                        adapter_segments,
                        dtype=torch.int32,
                        device=batch.adapter_meta.adapter_segments.device,
                    )
                prev_batches.append(
                    {
                        "next_token_ids": next_input_ids,
                        "next_token_logprobs": next_token_logprobs,
                        "accepted_ids": accepted_ids,
                    }
                )
                idx = len(prev_batches) - 1

                for req_idx, req in enumerate(batch.requests):
                    new_input_length = 1
                    if batch.speculative_logits is not None:
                        new_cache_length = (
                            batch.cache_lengths[req_idx]
                            + batch.input_lengths[req_idx]
                            + accepted_ids[req_idx]
                            - 1
                        )
                    else:
                        new_cache_length = (
                            batch.cache_lengths[req_idx] + batch.input_lengths[req_idx]
                        )
                    batch.cache_lengths[req_idx] = new_cache_length
                    batch.max_input_length = max(
                        batch.max_input_length, new_input_length
                    )
                    batch.input_lengths[req_idx] = new_input_length
                    current_length = new_cache_length + new_input_length
                    batch.max_current_length = max(
                        batch.max_current_length, current_length
                    )

                    requests_to_generate.append(
                        {
                            "idx": idx,
                            "request_id": req.id,
                            "prefix_offset": batch.prefix_offsets[req_idx],
                            "read_offset": batch.read_offsets[req_idx],
                            "stopping_criteria": batch.stopping_criterias[req_idx],
                            "all_input_ids": batch.all_input_ids[req_idx],
                            "do_sample": batch.next_token_chooser.do_sample[req_idx],
                            "seed": batch.next_token_chooser.seeds[req_idx],
                            "top_n_tokens": batch.top_n_tokens[req_idx],
                            "top_token_ids": batch_top_token_ids[req_idx],
                            "top_token_logprobs": batch_top_token_logprobs[req_idx],
                        }
                    )
                if prefill:
                    # We do not need prefill tensors anymore
                    batch.cu_seqlen_prefill = None
                    batch.prefill_cache_indices = None
                    batch.prefill_cu_outlens = None
                    batch.prefill_head_indices = None
                    batch.prefill_next_token_indices = None
                batch.next_token_logits = None
                batch.speculative_ids = None

        htorch.core.mark_step()
        # Stage 2. Prepare new batch for speculative scheduling
        if len(batches) > 1:
            if self.bucketing_ctx is not None:
                total_batch_size = 0
                for b in batches:
                    total_batch_size += len(b)
                padded_total_bs = self.bucketing_ctx.get_padded_decode_batch_size(
                    total_batch_size
                )
                batch = self.batch_type.concatenate(
                    batches, padded_total_bs=padded_total_bs
                )
            else:
                batch = self.batch_type.concatenate(batches)
        else:
            batch = batches[0]
        prefill = batch.prefilling
        if prefill:
            if self.bucketing_ctx is not None:
                batch.prepare_for_prefill(
                    self.bucketing_ctx.get_padded_prompt_seq_len(
                        batch.max_input_length
                    ),
                    self.bucketing_ctx.get_padded_prompt_batch_size(len(batch)),
                    self.max_total_tokens,
                    self.tokenizer.pad_token_id,
                )
            else:
                batch.prepare_for_prefill(
                    batch.max_input_length,
                    len(batch),
                    self.max_total_tokens,
                    self.tokenizer.pad_token_id,
                )
        else:
            batch.prepare_for_decode(
                self.dtype,
                self.use_contiguous_pa,
                self.bucketing_ctx,
                self.tokenizer.pad_token_id,
                self.sliding_window,
            )
        if hasattr(self, "set_inputs_embeds") and callable(self.set_inputs_embeds):
            self.set_inputs_embeds(batch)
        prefill_logprobs = batch.prefill_next_token_indices is not None
        # Update adapter indices for speculative tokens (if present)
        adapter_meta = batch.adapter_meta
        if adapter_meta is not None:
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
        else:
            adapter_data = None

        out, speculative_logits = self.forward(batch, adapter_data)

        if prefill:
            batch.next_token_logits = (
                out[batch.prefill_next_token_indices] if prefill_logprobs else out
            )
            if speculative_logits is not None:
                speculative_logits = (
                    speculative_logits[batch.prefill_next_token_indices]
                    if prefill_logprobs
                    else speculative_logits
                )
        else:
            prefill_logprobs = None
            batch.next_token_logits = out
        batch.speculative_logits = speculative_logits

        # HPU->CPU sync
        htorch.core.mark_step()
        start_decode = time.time_ns()
        for prev_batch in prev_batches:
            prev_batch["next_token_logprobs"] = prev_batch[
                "next_token_logprobs"
            ].tolist()
            prev_batch["next_token_ids"] = prev_batch["next_token_ids"].tolist()
            prev_batch["accepted_ids"] = prev_batch["accepted_ids"].tolist()
        htorch.core.mark_step()
        # Stage 3. Finish and return previous generations
        # Results
        generations: List[Generation] = []
        stopped = len(requests_to_generate) > 0
        # Reset max_input_length
        batch.max_input_length = 0
        # For each member of the batch
        indexs = [0] * len(prev_batches)
        idx_accept_ids = [0] * len(prev_batches)
        for i, req_data in enumerate(requests_to_generate):
            idx = req_data["idx"]
            request_id = req_data["request_id"]
            prefix_offset = req_data["prefix_offset"]
            read_offset = req_data["read_offset"]
            stopping_criteria = req_data["stopping_criteria"]
            all_input_ids = req_data["all_input_ids"]
            do_sample = req_data["do_sample"]
            seed = req_data["seed"]
            top_n_tokens = req_data["top_n_tokens"]
            n_accepted_ids = prev_batches[idx]["accepted_ids"][idx_accept_ids[idx]]
            top_token_ids = req_data["top_token_ids"]
            top_token_logprobs = req_data["top_token_logprobs"]
            # Append next token to all tokens
            next_token_texts = []
            left = 0

            if n_accepted_ids > 1:
                log_master(logger.debug, f"speculated ids {n_accepted_ids - 1}")

            current_stopped = False
            index = indexs[idx]
            for j in range(index, index + n_accepted_ids):
                # Generated token
                next_token_id = prev_batches[idx]["next_token_ids"][j]
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

            _next_token_ids = prev_batches[idx]["next_token_ids"][
                index : index + n_accepted_ids - left
            ]
            _next_token_logprobs = prev_batches[idx]["next_token_logprobs"][
                index : index + n_accepted_ids - left
            ]

            # Shard generations
            # All generations will be appended in the rust sharded client
            if request_id % self.world_size == self.rank:
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
                    request_id,
                    None,
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
            indexs[idx] += n_accepted_ids
            idx_accept_ids[idx] += 1

            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids
        htorch.core.mark_step()
        if stopped:
            # No need to return a batch if we know that all requests stopped
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)
