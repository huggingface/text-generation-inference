# Origin:   https://github.com/predibase/lorax
# Path:     lorax/server/lorax_server/utils/sgmv.py
# License:  Apache License Version 2.0, January 2004

import os
import warnings
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn.functional as F

try:
    import punica_kernels as _kernels

    HAS_SGMV = not bool(os.environ.get("DISABLE_SGMV", ""))
except ImportError:
    warnings.warn("Could not import SGMV kernel from Punica, falling back to loop.")
    _kernels = None
    HAS_SGMV = False


MIN_SGMV_RANK = 8
MIN_RANK_CUSTOM = 16
MAX_RANK_CUSTOM = 128
SGMV_BLOCK_SIZE = 16
BGMV_MAX_RANK = 64


def has_sgmv() -> bool:
    return HAS_SGMV


def pad_rank(t: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """Pad a tensor to the minimum rank for SGMV and the nearest multiple of the SGMV block size."""
    if not has_sgmv():
        return t

    # tensor parallelism will result in effective rank being divided by world_size,
    # so we need to scale the min rank to offset that effect
    min_rank = MIN_SGMV_RANK * world_size

    # if we're at or below the min rank, pad up to the min rank
    # otherwise, pad to the nearest multiple of the block size
    current_rank = t.size(dim)
    target_rank = (
        min_rank
        if current_rank <= min_rank
        else (current_rank + SGMV_BLOCK_SIZE - 1) // SGMV_BLOCK_SIZE * SGMV_BLOCK_SIZE
    )
    if current_rank == target_rank:
        return t

    pad_size = target_rank - current_rank

    # see complicatd pad syntax here: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    pad = [0, 0] * t.dim()
    pad[(t.dim() - dim - 1) * 2 + 1] = pad_size
    pad = tuple(pad)

    return F.pad(t, pad, mode="constant", value=0.0)


def use_cutlass_shrink(lora_rank: int) -> bool:
    return lora_rank < MIN_RANK_CUSTOM


def orient_for_rank(t: torch.Tensor, rank: int) -> torch.Tensor:
    if MIN_RANK_CUSTOM <= rank <= MAX_RANK_CUSTOM:
        return t.transpose(0, 1)
    return t


# Source: https://github.com/punica-ai/punica/blob/master/src/punica/ops/__init__.py
def add_lora_sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s_start: torch.Tensor,
    s_end: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
    Semantics:
        y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]).T @ deref(wb_ptr[i])

    Args:
        y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
        x: Shape: `[B, H1]`. Input vectors.
        wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
            Weight matrix shape: `[num_layers, R, H1]`.
        wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
            Weight matrix shape: `[num_layers, R, H2]`.
        s_start: Shape: `[S]`, DType: torch.int32. Indptr of the weight matrices start indices.
        s_end: Shape: `[S]`, DType: torch.int32. Indptr of the weight matrices end indices.
        layer_idx: Layer index of the weight matrices.
    """
    if lora_rank < MIN_RANK_CUSTOM or lora_rank > MAX_RANK_CUSTOM:
        # Custom SGMV shrink only supports rank 16, 32, 64, 128
        _add_lora_sgmv_cutlass_legacy(
            y, x, wa_ptr, wb_ptr, s_start, s_end, layer_idx, lora_rank
        )
        return

    tmp1 = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
    tmp2_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp2 = torch.empty((tmp2_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_shrink(v, x, wa_ptr, s_start, s_end, tmp1, layer_idx)
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, tmp2, layer_idx)


def _add_lora_sgmv_cutlass_legacy(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
    lora_rank: int,
):
    tmp_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_cutlass(v, x, wa_ptr, s_start, s_end, tmp, layer_idx)
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, tmp, layer_idx)


@lru_cache(maxsize=1)
def get_tmp_tensor(device: torch.device) -> torch.Tensor:
    return torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=device)


@lru_cache(maxsize=32)
def get_tmp_tensor_for_size(size: int, device: torch.device) -> torch.Tensor:
    tmp_size = _kernels.sgmv_cutlass_tmp_size(size)
    return torch.empty((tmp_size,), dtype=torch.uint8, device=device)


def get_tmp_tensor_for_size_no_kernels(size: int, device: torch.device) -> torch.Tensor:
    return torch.empty((size,), dtype=torch.uint8, device=device)


def get_tmp_expand_size(size: int) -> int:
    return _kernels.sgmv_cutlass_tmp_size(size)


def get_tmp_tensors(
    nsegments: int, lora_rank: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    use_cutlass = use_cutlass_shrink(lora_rank) and has_sgmv()
    has_sgmv_available = has_sgmv()

    if use_cutlass:
        tmp = get_tmp_tensor_for_size(nsegments, device)
        return tmp, tmp
    elif has_sgmv_available:
        return get_tmp_tensor(device), get_tmp_tensor_for_size(nsegments, device)
    else:
        tmp = get_tmp_tensor_for_size(nsegments, device)
        return tmp, tmp


def lora_a_sgmv_cutlass(
    x: torch.Tensor,
    tmp: torch.Tensor,
    wa_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
    lora_rank: int,
) -> torch.Tensor:
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    if MIN_RANK_CUSTOM <= lora_rank <= MAX_RANK_CUSTOM:
        _kernels.sgmv_shrink(v, x, wa_ptr, s_start, s_end, tmp, layer_idx)
    else:
        _kernels.sgmv_cutlass(v, x, wa_ptr, s_start, s_end, tmp, layer_idx)
    return v


def lora_b_sgmv_cutlass(
    y: torch.Tensor,
    v: torch.Tensor,
    tmp: torch.Tensor,
    wb_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
):
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, tmp, layer_idx)


"""
Semantics:
    y[i] += (
        x[i].unsqueeze(0)
        @ wa_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        @ wb_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        * scale
    ).squeeze(0)

Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    v: Shape: `[B, R]`. Temporary vector.
    x: Shape: `[B, H1]`. Input vectors.
    wa_T_all: Shape: `[None, L, R, H1]`. All of the transposed LoRA A matrices.
    wb_T_all: Shape: `[None, L, H2, R]`. All of the transposed LoRA B matrices.
    indicies: Shape: `[B]`. Indices of the LoRA weights.
    layer_idx: Layer index of LoRA weights.
    scale: Scaling factor.
"""


def add_lora_a_bgmv(
    v: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
):
    _kernels.dispatch_bgmv(v, x, wa_T_all, indicies, layer_idx, 1.0)


def add_lora_b_bgmv(
    y: torch.Tensor,
    v: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
):
    _kernels.dispatch_bgmv(y, v, wb_T_all, indicies, layer_idx, 1.0)


def segmented_matmul(
    y: torch.Tensor,
    x: torch.Tensor,
    w: List[torch.Tensor],
    b: List[torch.Tensor],
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
):
    for i in range(len(w)):
        if s_end[i] - s_start[i] <= 0:
            continue

        xi = x[s_start[i] : s_end[i]]
        wi = w[i]
        bi = b[i]
        y[s_start[i] : s_end[i]] = F.linear(xi, wi, bi)
