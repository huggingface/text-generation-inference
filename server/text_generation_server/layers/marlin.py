from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from text_generation_server.utils.weights import Weights, WeightsLoader
import torch
import torch.nn as nn

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


GPTQ_MARLIN_BITS = [4, 8]
GPTQ_MARLIN_GROUP_SIZES = [-1, 32, 64, 128]
MARLIN_TILE_SIZE = 16


class MarlinWeightsLoader(WeightsLoader):
    """Loader for Marlin-quantized weights."""

    def __init__(self, *, bits: int, is_marlin_24: bool):
        self.bits = bits
        self.is_marlin_24 = is_marlin_24

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        if self.is_marlin_24:
            B = weights.get_packed_sharded(
                f"{prefix}.B_24", dim=1, block_sizes=block_sizes
            )
            B_meta = weights.get_packed_sharded(
                f"{prefix}.B_meta", dim=1, block_sizes=block_sizes
            )
            s = weights.get_packed_sharded(
                f"{prefix}.s", dim=1, block_sizes=block_sizes
            )

            weight = GPTQMarlin24Weight(B=B, B_meta=B_meta, s=s, bits=self.bits)
        else:
            B = weights.get_packed_sharded(
                f"{prefix}.B", dim=1, block_sizes=block_sizes
            )
            s = weights.get_packed_sharded(
                f"{prefix}.s", dim=1, block_sizes=block_sizes
            )
            weight = MarlinWeight(B=B, s=s)

        return weight

    def get_multi_weights_col(self, weights: Weights, prefixes: List[str], dim: int):
        is_marlin_24 = getattr(self, "gptq_checkpoint_format", None) == "marlin_24"
        if is_marlin_24:
            try:
                B = torch.cat(
                    [weights.get_sharded(f"{p}.B_24", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `marlin` weight, make sure the model is already quantized"
                )

            B_meta = torch.cat(
                [weights.get_sharded(f"{p}.B_meta", dim=1) for p in prefixes], dim=1
            )

            s = torch.cat(
                [weights.get_sharded(f"{p}.s", dim=1) for p in prefixes], dim=1
            )

            weight = GPTQMarlin24Weight(B=B, B_meta=B_meta, s=s, bits=self.bits)
        else:
            try:
                B = torch.cat(
                    [weights.get_sharded(f"{p}.B", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `marlin` weight, make sure the model is already quantized"
                )
            s = torch.cat(
                [weights.get_sharded(f"{p}.s", dim=1) for p in prefixes], dim=1
            )

            weight = MarlinWeight(B=B, s=s)

        return weight

    def get_weights_row(self, weights: Weights, prefix: str):
        is_marlin_24 = getattr(self, "gptq_checkpoint_format", None) == "marlin_24"
        if is_marlin_24:
            try:
                B = weights.get_sharded(f"{prefix}.B_24", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `marlin` 2:4 sparsity weight, make sure the model is already quantized."
                )

            B_meta = weights.get_sharded(f"{prefix}.B_meta", dim=0)
            num_groups = weights._get_slice(f"{prefix}.s").get_shape()[0]
            if num_groups == 1:
                # The number of groups is 1 when groupsize == -1. share
                # scales between all shards in this case.
                s = weights.get_tensor(f"{prefix}.s")
            else:
                s = weights.get_sharded(f"{prefix}.s", dim=0)

            weight = GPTQMarlin24Weight(B=B, B_meta=B_meta, s=s, bits=self.bits)
        else:
            try:
                B = weights.get_sharded(f"{prefix}.B", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `marlin` weight, make sure the model is already quantized."
                )

            num_groups = weights._get_slice(f"{prefix}.s").get_shape()[0]
            if num_groups == 1:
                # The number of groups is 1 when groupsize == -1. share
                # scales between all shards in this case.
                s = weights.get_tensor(f"{prefix}.s")
            else:
                s = weights.get_sharded(f"{prefix}.s", dim=0)
            weight = MarlinWeight(B=B, s=s)

        return weight


def can_use_gptq_marlin(
    *, bits: int, groupsize: int, quant_method: str, quantize: str, sym: bool
) -> bool:
    return (
        SYSTEM == "cuda"
        and marlin_kernels is not None
        and has_sm_8_0
        and quantize == "gptq"
        and quant_method == "gptq"
        and bits in GPTQ_MARLIN_BITS
        and groupsize in GPTQ_MARLIN_GROUP_SIZES
        and sym
    )


def _check_marlin_kernels():
    if not (SYSTEM == "cuda" and has_sm_8_0):
        raise NotImplementedError(
            "Using quantized Marlin models requires a GPU with CUDA capability 8.0 or later."
        )

    if marlin_kernels is None:
        raise NotImplementedError(
            "marlin is not installed, install it with: pip install server/marlin"
        )


def _check_valid_shape(in_features: int, out_features: int):
    if (in_features % 128 != 0 or out_features % 64 != 0) and (
        in_features % 64 != 0 or out_features % 128 != 0
    ):
        raise ValueError(
            f"The GPTQ Marlin kernel does not have a valid thread configuration for weight matrix with shape ({out_features}, {in_features})."
            " The shape elements must be divisible by (128, 64) or (64, 128)."
        )


# https://github.com/IST-DASLab/marlin/blob/2f6d7c10e124b3c5fa29ff8d77d568bd7af3274c/marlin/__init__.py#L40C1-L68C54
def _get_perms() -> Tuple[List[int], List[int]]:
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


_scale_perm, _scale_perm_single = _get_perms()


def permute_scales(scales: torch.Tensor):
    out_features = scales.shape[1]
    if scales.shape[0] == 1:
        scales = scales.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    else:
        scales = scales.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    return scales.reshape((-1, out_features)).contiguous()


@dataclass
class GPTQMarlinWeight:
    """
    Repacked GPTQ Marlin weights.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor
    perm: torch.Tensor
    bits: int
    is_full_k: bool

    def __post_init__(self):
        assert self.qweight.dtype == torch.int32
        assert self.scales.dtype == torch.float16
        assert self.g_idx.dtype == torch.int32
        assert self.perm.dtype == torch.int32


def repack_gptq_for_marlin(
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    desc_act: bool,
    groupsize: int,
    sym: bool,
    sharded_infeatures: bool,
) -> GPTQMarlinWeight:
    """Convert GPTQ weights to a layout that's compatible with GPTQ-Marlin kernels."""
    _check_marlin_kernels()
    assert marlin_kernels is not None

    if bits not in GPTQ_MARLIN_BITS:
        supported_bits = ", ".join(str(b) for b in GPTQ_MARLIN_BITS)
        raise RuntimeError(
            f"Repacking {bits}-bit GPTQ weights as Marlin is not supported, must be one of: {supported_bits}"
        )

    if groupsize not in GPTQ_MARLIN_GROUP_SIZES:
        supported_sizes = ", ".join(str(b) for b in GPTQ_MARLIN_GROUP_SIZES)
        raise RuntimeError(
            f"Repacking GPTQ weights with group size {groupsize} as Marlin is not supported, must be one of: {supported_sizes}"
        )
    if not sym:
        raise RuntimeError(
            "Repacking GPTQ weights with asymmetric quantization as Marlin is not supported."
        )

    weights_per_int = 32 // bits
    in_features = qweight.shape[0] * weights_per_int
    out_features = qweight.shape[1]

    if in_features % groupsize != 0:
        raise ValueError(
            f"Number of input features ({in_features}) not divisible by group size ({groupsize})"
        )

    if desc_act and groupsize != -1:
        perm = torch.argsort(g_idx).to(torch.int)
        g_idx = g_idx[perm]
    else:
        perm = torch.empty(0, dtype=torch.int, device=qweight.device)
        g_idx = torch.empty(0, dtype=torch.int, device=qweight.device)

    repacked = marlin_kernels.gptq_marlin_repack(
        qweight, perm, in_features, out_features, bits
    )

    scales = permute_scales(scales)

    is_full_k = not (desc_act and sharded_infeatures)

    return GPTQMarlinWeight(
        qweight=repacked,
        scales=scales,
        g_idx=g_idx,
        perm=perm,
        bits=bits,
        is_full_k=is_full_k,
    )


class GPTQMarlinLinear(nn.Module):
    """
    Linear layer for GPTQ weights that were converted for the GPTQ-Marlin
    kernels.
    """

    def __init__(
        self,
        *,
        weight: GPTQMarlinWeight,
        bias: Optional[torch.Tensor],
    ):
        super().__init__()

        _check_marlin_kernels()
        assert marlin_kernels is not None

        in_features = weight.qweight.shape[0] * MARLIN_TILE_SIZE
        out_features = weight.scales.shape[1]
        _check_valid_shape(in_features=in_features, out_features=out_features)

        self.bits = weight.bits
        self.is_full_k = weight.is_full_k

        self.qweight = weight.qweight
        self.scales = weight.scales
        self.g_idx = weight.g_idx
        self.perm = weight.perm
        if bias is not None:
            self.bias = bias
        else:
            self.bias = None

        self.workspace = torch.zeros(
            out_features // 64 * 16, dtype=torch.int, device=weight.qweight.device
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        assert marlin_kernels is not None

        A_flat = A.view(-1, A.shape[-1])
        C = marlin_kernels.gptq_marlin_gemm(
            A_flat,
            self.qweight,
            self.scales,
            self.g_idx,
            self.perm,
            self.workspace,
            self.bits,
            A_flat.shape[0],
            self.scales.shape[1],
            A_flat.shape[1],
            self.is_full_k,
        )
        C = C.reshape(A.shape[:-1] + (self.scales.shape[1],))

        if self.bias is not None:
            C += self.bias

        return C


GPTQ_MARLIN_24_MIN_THREAD_N = 128
GPTQ_MARLIN_24_MIN_THREAD_K = 128
GPTQ_MARLIN_24_MAX_PARALLEL = 64
GPTQ_MARLIN_24_SUPPORTED_NUM_BITS = [4, 8]
GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES = [-1, 128]


@dataclass
class GPTQMarlin24Weight:
    """
    GPTQ-Marlin 2:4 weights.

    Attributes:
        B (torch.Tensor): int4-quantized weights packed into int32.
        B_meta (torch.Tensor): metadata for 2:4 sparsity.
        s (torch.Tensor): float16 scales.
        bits: quantized weight size.
    """

    B: torch.Tensor
    B_meta: torch.Tensor
    s: torch.Tensor
    bits: int

    def __post_init__(self):
        assert self.B.dtype == torch.int32
        assert self.B_meta.dtype == torch.int16
        assert self.s.dtype == torch.float16


class GPTQMarlin24Linear(nn.Module):
    def __init__(self, *, weight: GPTQMarlin24Weight, bias: Optional[torch.Tensor]):
        super().__init__()

        _check_marlin_kernels()
        assert marlin_kernels is not None

        if weight.bits not in GPTQ_MARLIN_BITS:
            supported_bits = ", ".join(str(b) for b in GPTQ_MARLIN_BITS)
            raise RuntimeError(
                f"{weight.bits}-bit GPTQ Sparse 2:4 Marlin is not supported, must be one of: {supported_bits}"
            )

        in_features = weight.B.shape[0] * MARLIN_TILE_SIZE * 2
        out_features = weight.s.shape[1]
        groupsize = -1 if weight.s.shape[0] == 1 else in_features // weight.s.shape[0]

        if groupsize not in GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES:
            supported_sizes = ", ".join(
                str(b) for b in GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES
            )
            raise RuntimeError(
                f"Group size {groupsize} is not supported, must be one of: {supported_sizes}"
            )

        self.bits = weight.bits
        weights_per_int32 = 32 // self.bits

        assert (
            out_features % GPTQ_MARLIN_24_MIN_THREAD_N == 0
        ), f"Number of output features ({out_features}) not divisable by {GPTQ_MARLIN_24_MIN_THREAD_N} threads"
        assert (
            out_features % weights_per_int32 == 0
        ), f"Number of output features ({out_features}) not divisable by weights per int32 ({weights_per_int32})"

        assert (
            in_features % GPTQ_MARLIN_24_MIN_THREAD_K == 0
        ), f"Number of output features ({out_features}) not divisable by {GPTQ_MARLIN_24_MIN_THREAD_K} threads"
        if groupsize != -1 and in_features % groupsize != 0:
            raise ValueError(
                f"Number of input features ({in_features}) not divisable by group size ({groupsize})"
            )

        self.B = weight.B
        self.B_meta = weight.B_meta
        self.s = weight.s
        if bias is not None:
            self.bias = bias
        else:
            self.bias = None

        self.workspace = torch.zeros(
            (out_features // GPTQ_MARLIN_24_MIN_THREAD_N) * GPTQ_MARLIN_24_MAX_PARALLEL,
            dtype=torch.int,
            device=weight.B.device,
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        assert marlin_kernels is not None

        C = marlin_kernels.gptq_marlin_24_gemm(
            A.view(-1, A.shape[-1]),
            self.B,
            self.B_meta,
            self.s,
            self.workspace,
            self.bits,
            A.shape[0],
            self.s.shape[1],
            A.shape[1],
        )

        C = C.reshape(A.shape[:-1] + (self.s.shape[1],))

        if self.bias is not None:
            C += self.bias

        return C


@dataclass
class MarlinWeight:
    """
    Marlin weights.

    Attributes:
        B (torch.Tensor): int4-quantized weights packed into int32.
        s (torch.Tensor): float16 scales.
    """

    B: torch.Tensor
    s: torch.Tensor

    def __post_init__(self):
        assert self.B.dtype == torch.int32
        assert self.s.dtype == torch.float16


class MarlinLinear(nn.Module):
    def __init__(self, *, weight: MarlinWeight, bias: Optional[torch.Tensor]):
        super().__init__()

        _check_marlin_kernels()
        assert marlin_kernels is not None

        in_features = weight.B.shape[0] * MARLIN_TILE_SIZE
        out_features = weight.s.shape[1]
        assert (
            in_features % 128 == 0
        ), f"Number of input features ({in_features}) not divisable by 128"
        assert (
            out_features % 256 == 0
        ), f"Number of output features ({out_features}) not divisable by 256"

        groupsize = -1 if weight.s.shape[0] == 1 else in_features // weight.s.shape[0]
        assert groupsize in {
            -1,
            128,
        }, f"Group size must be -1 or 128, was {groupsize}"

        self.B = weight.B
        self.s = weight.s
        if bias is not None:
            self.bias = bias
        else:
            self.bias = None

        self.workspace = torch.zeros(
            out_features // 64 * 16, dtype=torch.int, device=weight.B.device
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        assert marlin_kernels is not None

        C = marlin_kernels.marlin_gemm(
            A.view(-1, A.shape[-1]),
            self.B,
            self.s,
            self.workspace,
            A.shape[0],
            self.s.shape[1],
            A.shape[1],
        )
        C = C.reshape(A.shape[:-1] + (self.s.shape[1],))

        if self.bias is not None:
            C += self.bias

        return C
