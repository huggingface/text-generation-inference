from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from text_generation_server.layers.marlin.util import _check_marlin_kernels
from text_generation_server.utils.weights import Weight, Weights, WeightsLoader

try:
    import marlin_kernels
except ImportError:
    marlin_kernels = None


class MarlinWeightsLoader(WeightsLoader):
    """Loader for Marlin-quantized weights."""

    def __init__(self, *, bits: int, is_marlin_24: bool):
        self.bits = bits
        self.is_marlin_24 = is_marlin_24

    def get_weights(self, weights: "Weights", prefix: str):
        """
        Get weights at the given prefix and apply without tensor paralllism.
        """
        is_marlin_24 = getattr(self, "gptq_checkpoint_format", None) == "marlin_24"
        if is_marlin_24:
            try:
                B = weights.get_tensor(f"{prefix}.B_24")
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `marlin` 2:4 sparsity weight, make sure the model is already quantized."
                )

            B_meta = weights.get_tensor(f"{prefix}.B_meta")
            s = weights.get_tensor(f"{prefix}.s")
            weight = GPTQMarlin24Weight(B=B, B_meta=B_meta, s=s, bits=self.bits)
        else:
            try:
                B = weights.get_tensor(f"{prefix}.B")
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `marlin` weight, make sure the model is already quantized."
                )

            s = weights.get_tensor(f"{prefix}.s")
            weight = MarlinWeight(B=B, s=s)

        return weight

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
        if self.is_marlin_24:
            try:
                B = torch.cat(
                    [weights.get_sharded(f"{p}.B_24", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `marlin` weight, make sure the model is already quantized"
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
                    "Cannot load `marlin` weight, make sure the model is already quantized"
                )
            s = torch.cat(
                [weights.get_sharded(f"{p}.s", dim=1) for p in prefixes], dim=1
            )

            weight = MarlinWeight(B=B, s=s)

        return weight

    def get_weights_row(self, weights: Weights, prefix: str):
        if self.is_marlin_24:
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


@dataclass
class MarlinWeight(Weight):
    """
    Marlin weights.

    Attributes:
        B (torch.Tensor): int4-quantized weights packed into int32.
        s (torch.Tensor): bfloat16/float16 scales.
    """

    B: torch.Tensor
    s: torch.Tensor

    def __post_init__(self):
        assert self.B.dtype == torch.int32
        assert self.s.dtype in [torch.float16, torch.bfloat16]

    def get_linear(self, bias: torch.Tensor):
        return MarlinLinear(weight=self, bias=bias)


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


GPTQ_MARLIN_24_MIN_THREAD_N = 128
GPTQ_MARLIN_24_MIN_THREAD_K = 128
GPTQ_MARLIN_24_MAX_PARALLEL = 64
GPTQ_MARLIN_24_SUPPORTED_NUM_BITS = [4, 8]
GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES = [-1, 128]
MARLIN_TILE_SIZE = 16


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

    def get_linear(self, bias: torch.Tensor):
        return GPTQMarlin24Linear(
            weight=self,
            bias=bias,
        )


class GPTQMarlin24Linear(nn.Module):
    def __init__(self, *, weight: GPTQMarlin24Weight, bias: Optional[torch.Tensor]):
        super().__init__()

        _check_marlin_kernels()
        assert marlin_kernels is not None

        if weight.bits not in GPTQ_MARLIN_24_SUPPORTED_NUM_BITS:
            supported_bits = ", ".join(
                str(b) for b in GPTQ_MARLIN_24_SUPPORTED_NUM_BITS
            )
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
