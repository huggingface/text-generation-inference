from typing import Optional
from contextvars import ContextVar
from contextlib import contextmanager

import flashinfer
import torch

prefill_state: ContextVar[flashinfer.BatchPrefillWithRaggedKVCacheWrapper] = ContextVar(
    "prefill_state"
)

prefill_with_paged_kv_state: ContextVar[
    flashinfer.BatchPrefillWithPagedKVCacheWrapper
] = ContextVar("prefill_with_paged_kv_state")

decode_state: ContextVar[flashinfer.BatchDecodeWithPagedKVCacheWrapper] = ContextVar(
    "decode_state"
)

workspace: Optional[torch.Tensor] = None


def get_workspace(device):
    """Get shared flashinfer workspace."""
    global workspace
    if workspace is None:
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return workspace


def create_prefill_with_paged_kv_state(
    *,
    device: torch.device,
):
    """Create a prefill state that uses the KV cache."""
    workspace_buffer = get_workspace(device)
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_cuda_graph=False
    )


@contextmanager
def use_prefill_with_paged_kv_state(
    *,
    state: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    block_tables: torch.Tensor,
    cu_seqlens: torch.Tensor,
    input_lengths: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    dtype: torch.dtype,
    window_left: int,
):
    """
    Context manager to set the active flashinfer prefill state to the given
    `state` and parameters. This state will be used by all calls to the
    `attention` function while the context manager is active.
    """

    indptr = torch.zeros(
        input_lengths.shape[0] + 1, device=input_lengths.device, dtype=torch.int32
    )
    # Round up to page size and then calculate the cumulative sum to get
    # the indices into the block table.
    torch.add(input_lengths, page_size - 1, out=indptr[1:])
    indptr[1:].div_(page_size, rounding_mode="floor")
    indptr[1:].cumsum_(-1)

    # Get the lengths of the last page in a block.
    if page_size == 1:
        last_page_len = torch.ones(
            input_lengths.shape[0], dtype=torch.int32, device=input_lengths.device
        )
    else:
        last_page_len = torch.empty(
            input_lengths.shape[0], dtype=torch.int32, device=input_lengths.device
        )
        torch.sub(input_lengths, 1, out=last_page_len)
        last_page_len.remainder_(page_size)
        last_page_len += 1

    token = prefill_with_paged_kv_state.set(state)
    try:
        state.begin_forward(
            qo_indptr=cu_seqlens,
            paged_kv_indptr=indptr,
            paged_kv_indices=block_tables,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            q_data_type=dtype,
            page_size=page_size,
            window_left=window_left,
        )
        yield
    finally:
        state.end_forward()
        if token is not None:
            prefill_with_paged_kv_state.reset(token)


def create_prefill_state(
    *,
    device: torch.device,
):
    """Create a prefill state."""
    workspace_buffer = get_workspace(device)
    return flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_cuda_graph=False
    )


@contextmanager
def use_prefill_state(
    *,
    state: flashinfer.BatchPrefillWithRaggedKVCacheWrapper,
    cu_seqlens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    window_left: int,
):
    """
    Context manager to set the active flashinfer prefill state to the given
    `state` and parameters. This state will be used by all calls to the
    `attention` function while the context manager is active.
    """

    token = prefill_state.set(state)
    try:
        state.begin_forward(
            qo_indptr=cu_seqlens,
            kv_indptr=cu_seqlens,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            q_data_type=dtype,
            window_left=window_left,
        )
        yield
    finally:
        state.end_forward()
        if token is not None:
            prefill_state.reset(token)


def create_decode_state(
    *,
    device: torch.device,
    num_heads: int,
    num_kv_heads: int,
):
    """Create a decode state."""
    workspace_buffer = get_workspace(device)
    num_groups = num_heads // num_kv_heads
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=False,
        # Taken from https://github.com/flashinfer-ai/flashinfer/blob/33ef95700981ba70f4cab63b8931e562bc795b21/python/flashinfer/decode.py#L57-L60
        use_tensor_cores=num_groups not in [1, 2, 4, 8],
    )


def create_decode_state_cuda_graphs(
    *,
    device: torch.device,
    block_tables: torch.Tensor,
    block_tables_ptr: torch.Tensor,
    last_page_len: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
):
    """
    Create a decode state for use with CUDA Graphs. `block_tables`,
    `block_tables_ptr`, and `last_page_len` are used in CUDA Graphs and are
    therefore stored as part of the state.
    """
    workspace_buffer = get_workspace(device)
    num_groups = num_heads // num_kv_heads
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=True,
        paged_kv_indices_buffer=block_tables,
        paged_kv_indptr_buffer=block_tables_ptr,
        paged_kv_last_page_len_buffer=last_page_len,
        # Taken from https://github.com/flashinfer-ai/flashinfer/blob/33ef95700981ba70f4cab63b8931e562bc795b21/python/flashinfer/decode.py#L57-L60
        use_tensor_cores=num_groups not in [1, 2, 4, 8],
    )


@contextmanager
def use_decode_state(
    *,
    state: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
    input_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    kv_cache_dtype: torch.dtype,
    dtype: torch.dtype,
    window_left: int,
):
    """
    Context manager to set the active flashinfer decoding state to the given
    `state` and parameters. This state will be used by all calls to the
    `paged_attention` function while the context manager is active.
    """
    indptr = torch.zeros(
        input_lengths.shape[0] + 1, device=input_lengths.device, dtype=torch.int32
    )
    # Round up to page size and then calculate the cumulative sum to get
    # the indices into the block table.
    torch.add(input_lengths, page_size - 1, out=indptr[1:])
    indptr[1:].div_(page_size, rounding_mode="floor")
    indptr[1:].cumsum_(-1)

    # Get the lengths of the last page in a block.
    last_page_len = torch.empty(
        input_lengths.shape[0], dtype=torch.int32, device=input_lengths.device
    )
    torch.sub(input_lengths, 1, out=last_page_len)
    last_page_len.remainder_(page_size)
    last_page_len += 1

    token = decode_state.set(state)

    try:
        state.begin_forward(
            indptr=indptr,
            indices=block_tables,
            last_page_len=last_page_len,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            page_size=page_size,
            data_type=kv_cache_dtype,
            q_data_type=dtype,
            window_left=window_left,
        )
        yield
    finally:
        state.end_forward()
        if token is not None:
            decode_state.reset(token)
