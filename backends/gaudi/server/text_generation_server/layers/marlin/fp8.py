from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from text_generation_server.layers.fp8 import fp8_quantize
from text_generation_server.layers.marlin.gptq import _check_valid_shape
from text_generation_server.layers.marlin.util import (
    _check_marlin_kernels,
    permute_scales,
)
from text_generation_server.utils.log import log_once

try:
    import marlin_kernels
except ImportError:
    marlin_kernels = None


MARLIN_TILE_SIZE = 16


class GPTQMarlinFP8Linear(nn.Module):
    """
    FP8 GPTQ-Marlin linear layer.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> None:
        super().__init__()

        _check_marlin_kernels()
        assert marlin_kernels is not None

        log_once(logger.info, "GPU does not support FP8, using Marlin FP8 kernel")

        scales = scales.unsqueeze(0)
        if scales.shape[1] == 1:
            out_features, in_features = qweight.shape
            scales = scales.repeat(1, out_features)
        qweight, scales = repack_fp8_for_marlin(qweight, scales)

        in_features = qweight.shape[0] * MARLIN_TILE_SIZE
        out_features = scales.shape[1]
        _check_valid_shape(in_features=in_features, out_features=out_features)

        self.qweight = qweight
        self.scales = scales
        self.bias = bias if bias is not None else None

        self.workspace = torch.zeros(
            out_features // 64 * 16, dtype=torch.int, device=qweight.device
        )

    @classmethod
    def from_unquant(cls, weight, bias, dtype):
        qweight, scales = fp8_quantize(weight)
        return cls(qweight=qweight, scales=scales.to(dtype), bias=bias)

    @classmethod
    def from_fp8(cls, weight, scale, _input_scale, bias, dtype):
        return cls(qweight=weight, scales=scale.to(dtype), bias=bias)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        assert marlin_kernels is not None

        A_flat = A.view(-1, A.shape[-1])
        C = marlin_kernels.fp8_marlin_gemm(
            A_flat,
            self.qweight,
            self.scales,
            self.workspace,
            8,
            A_flat.shape[0],
            self.scales.shape[1],
            A_flat.shape[1],
        )
        C = C.reshape(A.shape[:-1] + (self.scales.shape[1],))

        if self.bias is not None:
            C += self.bias

        return C


def pack_fp8_as_int32(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements).
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn

    if fp8_tensor.shape[0] % 4 != 0:
        raise ValueError(
            f"Leading tensor dimension is not divisable by 4: {fp8_tensor.shape[0]}"
        )

    # Reshape to prepare for packing
    reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])

    # Convert fp8 to uint8 (byte) representation
    byte_tensor = reshaped.view(torch.uint8)

    # Pack 4 uint8 values into one int32
    packed = torch.zeros(
        fp8_tensor.shape[0] // 4,
        fp8_tensor.shape[1],
        dtype=torch.int32,
        device=fp8_tensor.device,
    )

    for i in range(4):
        packed.bitwise_or_(byte_tensor[:, i].to(torch.int32) << i * 8)

    return packed


def repack_fp8_for_marlin(weight: torch.Tensor, scales: torch.Tensor):
    """
    Repack FP8 tensor for GPTQ-Marlin.
    """

    out_features, in_features = weight.shape

    # Torch linear layers weights with shape [out_features, in_features],
    # GPTQ-quantized weights use [in_feateres/pack_factor, in_features],
    # so transpose before packing.
    qweight = pack_fp8_as_int32(weight.t())

    perm = torch.empty(0, dtype=torch.int, device=qweight.device)
    repacked = marlin_kernels.gptq_marlin_repack(
        qweight, perm, in_features, out_features, 8
    )

    scales = permute_scales(scales)

    return repacked, scales
