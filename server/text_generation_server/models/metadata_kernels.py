import torch
import triton

import triton.language as tl

from loguru import logger
from typing import List, Optional
from torch.utils._triton import has_triton as has_triton_torch

from text_generation_server.utils.import_utils import (
    SYSTEM,
)
from text_generation_server.utils.log import log_master

_HAS_TRITON: Optional[bool] = None


def has_triton():
    global _HAS_TRITON
    if _HAS_TRITON is None:
        # FIXME: it seems that has_triton_torch is bugged on RocM
        #        For now, only accept cuda
        _HAS_TRITON = has_triton_torch() if SYSTEM == "cuda" else False
        if _HAS_TRITON:
            log_master(logger.info, "Using optimized Triton indexing kernels.")

    return _HAS_TRITON


def block_tables_to_padded(
    max_blocks: int,
    cu_seqlen: torch.Tensor,
    block_tables: torch.Tensor,
    block_tables_ragged: torch.Tensor,
):
    def grid(meta):
        return (
            triton.cdiv(max_blocks, meta["BLOCK_SIZE"]),
            len(block_tables),
        )

    triton_block_tables_to_padded[grid](
        cu_seqlen,
        block_tables,
        block_tables_ragged,
        block_tables.shape[1],
        BLOCK_SIZE=256,
    )


def block_tables_to_ragged(
    *,
    block_tables: torch.Tensor,
    input_lengths: List[int],
    cache_lengths: List[int],
    input_lengths_tensor: torch.Tensor,
    cache_lengths_tensor: torch.Tensor,
    max_current_length: int
) -> torch.Tensor:
    """Convert block table to ragged format compatible with FlashInfer."""
    assert len(input_lengths) == len(cache_lengths)

    total_len = sum(input_lengths) + sum(cache_lengths)
    block_tables_ragged = torch.empty(
        total_len, dtype=torch.int32, device=block_tables.device
    )

    if has_triton():
        cu_seqlen = torch.nn.functional.pad(
            torch.cumsum(input_lengths_tensor + cache_lengths_tensor, dim=0), (1, 0)
        )

        def grid(meta):
            return (
                triton.cdiv(max_current_length, meta["BLOCK_SIZE"]),
                len(cache_lengths),
            )

        triton_block_tables_to_ragged[grid](
            cu_seqlen,
            block_tables,
            block_tables_ragged,
            block_tables.shape[1],
            BLOCK_SIZE=256,
        )
    else:
        offset = 0
        for i, (input_length, cache_length) in enumerate(
            zip(input_lengths, cache_lengths)
        ):
            seq_len = cache_length + input_length
            block_tables_ragged[offset : offset + seq_len] = block_tables[i][:seq_len]
            offset += seq_len

    return block_tables_ragged


def copy_next_input_ids_inplace(
    max_next_input_ids: int,
    all_input_ids: torch.Tensor,
    cache_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    next_input_ids: torch.Tensor,
    cu_accepted_ids: torch.Tensor,
):
    def grid(meta):
        return (
            triton.cdiv(max_next_input_ids, meta["BLOCK_SIZE"]),
            len(all_input_ids),
        )

    triton_copy_next_input_ids_inplace[grid](
        all_input_ids,
        cache_lengths,
        input_lengths,
        prompt_lengths,
        next_input_ids,
        cu_accepted_ids,
        all_input_ids.shape[1],
        BLOCK_SIZE=16,
    )


def prepare_position_slot_ids(
    max_input_length: int,
    cache_lengths: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cu_slots: torch.Tensor,
    position_ids: torch.Tensor,
    slot_indices: torch.Tensor,
):
    def grid(meta):
        return (
            triton.cdiv(max_input_length, meta["BLOCK_SIZE"]),
            len(cache_lengths),
        )

    triton_prepare_position_slot_ids[grid](
        cache_lengths, cu_seqlen, cu_slots, position_ids, slot_indices, BLOCK_SIZE=256
    )


def slots_filtering(
    max_slots: int,
    slots: torch.Tensor,
    filtered_slots: torch.Tensor,
    cu_slots: torch.Tensor,
    slots_start: torch.Tensor,
):
    def grid(meta):
        return (
            triton.cdiv(max_slots, meta["BLOCK_SIZE"]),
            len(slots_start),
        )

    triton_slots_filtering[grid](
        slots, filtered_slots, slots_start, cu_slots, BLOCK_SIZE=256
    )


@triton.jit
def triton_slots_filtering(
    # Inputs
    slots_ptr,
    filtered_slots_ptr,
    slots_start_ptr,
    cu_slots_ptr,
    # Const values
    BLOCK_SIZE: "tl.constexpr",
):
    # Position in block_tables_ragged.numel() / BLOCK_SIZE
    pid = tl.program_id(axis=0)
    # Position in batch
    bid = tl.program_id(axis=1)

    block_start = pid * BLOCK_SIZE
    block_arange = block_start + tl.arange(0, BLOCK_SIZE)

    filter_start = tl.load(slots_start_ptr + bid)

    slot_start = tl.load(cu_slots_ptr + bid)
    slot_end = tl.load(cu_slots_ptr + bid + 1)

    mask = (slot_start + block_arange) < slot_end

    slots = tl.load(slots_ptr + filter_start + block_arange, mask=mask)
    tl.store(filtered_slots_ptr + slot_start + block_arange, slots, mask=mask)


@triton.jit
def triton_block_tables_to_padded(
    # Inputs
    cu_seqlen_ptr,
    # Outputs
    block_tables_ptr,
    block_tables_ragged_ptr,
    # Stride
    stride_block_tables,
    # Const values
    BLOCK_SIZE: "tl.constexpr",
):
    # Position in block_tables_ragged.numel() / BLOCK_SIZE
    pid = tl.program_id(axis=0)
    # Position in batch
    bid = tl.program_id(axis=1)

    block_start = pid * BLOCK_SIZE
    block_arange = block_start + tl.arange(0, BLOCK_SIZE)

    seq_start = tl.load(cu_seqlen_ptr + bid)
    seq_end = tl.load(cu_seqlen_ptr + bid + 1)

    mask = (seq_start + block_arange) < seq_end

    blocks = tl.load(block_tables_ragged_ptr + seq_start + block_arange, mask=mask)
    tl.store(
        block_tables_ptr + bid * stride_block_tables + block_arange, blocks, mask=mask
    )


@triton.jit
def triton_block_tables_to_ragged(
    # Inputs
    cu_seqlen_ptr,
    # Outputs
    block_tables_ptr,
    block_tables_ragged_ptr,
    # Stride
    stride_block_tables,
    # Const values
    BLOCK_SIZE: "tl.constexpr",
):
    # Position in block_tables_ragged.numel() / BLOCK_SIZE
    pid = tl.program_id(axis=0)
    # Position in batch
    bid = tl.program_id(axis=1)

    block_start = pid * BLOCK_SIZE
    block_arange = block_start + tl.arange(0, BLOCK_SIZE)

    seq_start = tl.load(cu_seqlen_ptr + bid)
    seq_end = tl.load(cu_seqlen_ptr + bid + 1)

    mask = (seq_start + block_arange) < seq_end

    blocks = tl.load(
        block_tables_ptr + bid * stride_block_tables + block_arange, mask=mask
    )
    tl.store(block_tables_ragged_ptr + seq_start + block_arange, blocks, mask=mask)


@triton.jit
def triton_copy_next_input_ids_inplace(
    # Inputs
    all_input_ids_ptr,
    cache_lengths_ptr,
    input_lengths_ptr,
    prompt_lengths_ptr,
    next_input_ids_ptr,
    cu_accepted_ids_ptr,
    # Stride
    stride_all_input_ids,
    # Const values
    BLOCK_SIZE: "tl.constexpr",
):
    # Position in max_accepted_ids / BLOCK_SIZE
    pid = tl.program_id(axis=0)
    # Position in batch
    bid = tl.program_id(axis=1)

    block_start = pid * BLOCK_SIZE
    block_arange = block_start + tl.arange(0, BLOCK_SIZE)

    # Used for correctly indexing in all_input_ids
    cache_length = tl.load(cache_lengths_ptr + bid)
    input_length = tl.load(input_lengths_ptr + bid)
    prompt_length = tl.load(prompt_lengths_ptr + bid)

    # Start/End of next_input_ids for this request
    next_input_ids_start = tl.load(cu_accepted_ids_ptr + bid)
    next_input_ids_end = tl.load(cu_accepted_ids_ptr + bid + 1)

    # Mask values out of range
    mask = (next_input_ids_start + block_arange) < next_input_ids_end

    # Mask values for request still prefilling
    decode_mask = (cache_length + input_length + block_arange) >= prompt_length

    mask = mask & decode_mask

    # Load this request next input ids
    next_input_ids = tl.load(
        next_input_ids_ptr + next_input_ids_start + block_arange, mask=mask
    )

    # Store in all_input_ids, since it is a 2D tensor, apply stride * bid
    tl.store(
        all_input_ids_ptr
        + stride_all_input_ids * bid
        + cache_length
        + input_length
        + block_arange,
        next_input_ids,
        mask=mask,
    )


@triton.jit
def triton_prepare_position_slot_ids(
    # Inputs
    cache_lengths_ptr,
    cu_seqlen_ptr,
    cu_slots_ptr,
    # Outputs
    position_ids_ptr,
    slot_indices_ptr,
    # Const values
    BLOCK_SIZE: "tl.constexpr",
):
    # Position in max_input_length / BLOCK_SIZE
    pid = tl.program_id(axis=0)
    # Position in batch
    bid = tl.program_id(axis=1)

    block_start = pid * BLOCK_SIZE
    block_arange = block_start + tl.arange(0, BLOCK_SIZE)

    cache_length = tl.load(cache_lengths_ptr + bid)

    seq_start = tl.load(cu_seqlen_ptr + bid)
    seq_end = tl.load(cu_seqlen_ptr + bid + 1)

    slot_start = tl.load(cu_slots_ptr + bid)

    mask = (seq_start + block_arange) < seq_end

    tl.store(
        position_ids_ptr + seq_start + block_arange,
        cache_length + block_arange,
        mask=mask,
    )
    tl.store(
        slot_indices_ptr + seq_start + block_arange,
        slot_start + cache_length + block_arange,
        mask=mask,
    )
