import functools
from typing import List, Tuple

import numpy
import torch
from text_generation_server.utils.import_utils import SYSTEM

try:
    import marlin_kernels
except ImportError:
    marlin_kernels = None

try:
    major, _minor = torch.cuda.get_device_capability()
    has_sm_8_0 = major >= 8
except Exception:
    has_sm_8_0 = False


def _check_marlin_kernels():
    if not (SYSTEM == "cuda" and has_sm_8_0):
        raise NotImplementedError(
            "Using quantized Marlin models requires a GPU with CUDA capability 8.0 or later."
        )

    if marlin_kernels is None:
        raise NotImplementedError(
            "marlin is not installed, install it with: pip install server/marlin"
        )


# https://github.com/IST-DASLab/marlin/blob/2f6d7c10e124b3c5fa29ff8d77d568bd7af3274c/marlin/__init__.py#L40C1-L68C54
@functools.cache
def get_perms() -> Tuple[List[int], List[int]]:
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def permute_scales(scales: torch.Tensor):
    scale_perm, scale_perm_single = get_perms()
    out_features = scales.shape[1]
    if scales.shape[0] == 1:
        scales = scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    else:
        scales = scales.reshape((-1, len(scale_perm)))[:, scale_perm]
    return scales.reshape((-1, out_features)).contiguous()


# Functions below are from vLLM


def get_pack_factor(bits: int) -> int:
    if 32 % bits != 0:
        raise ValueError(f"Cannot {bits} bit values into uint32")
    return 32 // bits


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (
        size_k,
        size_n // pack_factor,
    ), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
        packed_q_w.shape, size_k, size_n, pack_factor
    )

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    scale_perm, _ = get_perms()
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp
