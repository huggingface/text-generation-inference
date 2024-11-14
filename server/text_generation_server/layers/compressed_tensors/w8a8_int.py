from typing import List, Optional, Union, TypeVar
from dataclasses import dataclass

from loguru import logger
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationType

from text_generation_server.layers.fp8 import _load_scalar_or_matrix_scale
from text_generation_server.utils.log import log_once
from text_generation_server.utils.weights import Weight, Weights, WeightsLoader

try:
    import marlin_kernels
except ImportError:
    marlin_kernels = None


class W8A8IntLoader(WeightsLoader):
    """
    Loader for w8a8 integer compressed-tensors parameters.
    """

    def __init__(
        self,
        *,
        input_args: Optional[QuantizationArgs],
        weight_args: QuantizationArgs,
    ):
        if weight_args.type != QuantizationType.INT and weight_args.num_bits != 8:
            raise ValueError(
                f"{type(self).__name__} only supports w8a8 int checkpoints"
            )

        if not weight_args.symmetric:
            raise ValueError("Checkpoints with asymmetric weights are not supported")

        self.load_weight_scale = not weight_args.dynamic

        if input_args is not None:
            static = not input_args.dynamic
            symmetric = input_args.symmetric
            self.load_input_scale = static
            self.load_input_zero_point = static and not symmetric
            self.input_symmetric = input_args.symmetric

            if static:
                # People shouldn't really use static input quantization,
                # the output is pretty bad.
                log_once(
                    logger.warning,
                    "Using W8A8 int with static input quantization results in large regressions in accuracy. Consider dynamic input quantization instead.",
                )
        else:
            self.load_input_scale = False
            self.load_input_zero_point = False
            self.input_symmetric = True

    def __str__(self) -> str:
        def scale_to_str(scale):
            return "static" if scale else "dynamic"

        def symmetric_to_sting(symmetric):
            return "symmetric" if symmetric else "asymmetric"

        return f"{self.__class__.__name__} (w8a8 int, input: {symmetric_to_sting(self.input_symmetric)}, {scale_to_str(self.load_input_scale)})"

    def get_weights(self, weights: "Weights", prefix: str):
        w = weights.get_tensor(f"{prefix}.weight", to_dtype=False)

        weight_scale = None
        if self.load_weight_scale:
            weight_scale = weights.get_tensor(
                f"{prefix}.weight_scale", to_dtype=False
            ).reshape(-1)

        input_scale = None
        if self.load_input_scale:
            input_scale = weights.get_tensor(
                f"{prefix}.input_scale", to_dtype=False
            ).reshape(-1)

        input_zero_point = None
        if self.load_input_zero_point:
            input_zero_point = _get_tensor_or_else(
                weights,
                f"{prefix}.input_zero_point",
                torch.zeros((1,), device=w.device, dtype=torch.int8),
            ).reshape(-1)

        return Int8Weight(
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            input_symmetric=self.input_symmetric,
            weight=w,
            weight_scale=weight_scale,
        )

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        w = weights.get_packed_sharded(
            f"{prefix}.weight", dim=0, block_sizes=block_sizes, to_dtype=False
        )

        weight_scale = None
        if self.load_weight_scale:
            weight_scale = weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)
            if weight_scale.numel() > 1:
                weight_scale = weights.get_packed_sharded(
                    f"{prefix}.weight_scale",
                    dim=0,
                    block_sizes=block_sizes,
                    to_dtype=False,
                )
            weight_scale = weight_scale.reshape(-1)

        input_scale = None
        if self.load_input_scale:
            input_scale = weights.get_tensor(
                f"{prefix}.input_scale", to_dtype=False
            ).reshape(-1)

        input_zero_point = None
        if self.load_input_zero_point:
            input_zero_point = _get_tensor_or_else(
                weights,
                f"{prefix}.input_zero_point",
                torch.zeros((1,), device=w.device, dtype=torch.int8),
            ).reshape(-1)

        return Int8Weight(
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            input_symmetric=self.input_symmetric,
            weight=w,
            weight_scale=weight_scale,
        )

    def get_multi_weights_col(self, weights: "Weights", prefixes: List[str], dim: int):
        w = [
            weights.get_sharded(f"{p}.weight", dim=0, to_dtype=False) for p in prefixes
        ]
        shapes = [x.shape for x in w]

        w = torch.cat(w, dim=dim)

        weight_scale = None
        if self.load_weight_scale:
            weight_scale = [
                _load_scalar_or_matrix_scale(weights, f"{p}.weight_scale", shape)
                for p, shape in zip(prefixes, shapes)
            ]
            weight_scale = torch.cat(weight_scale, dim=0).reshape(-1, 1)

        input_scale = None
        if self.load_input_scale:
            input_scale = [
                weights.get_tensor(f"{p}.input_scale", to_dtype=False) for p in prefixes
            ]
            input_scale = torch.cat(input_scale, dim=0)

        input_zero_point = None
        if self.load_input_zero_point:
            input_zero_point = [
                _get_tensor_or_else(
                    weights,
                    f"{prefix}.input_zero_point",
                    torch.zeros((1,), device=w.device, dtype=torch.int8),
                )
                for prefix in prefixes
            ]
            input_zero_point = torch.cat(input_zero_point, dim=0)

        return Int8Weight(
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            input_symmetric=self.input_symmetric,
            weight=w,
            weight_scale=weight_scale,
        )

    def get_weights_row(self, weights: "Weights", prefix: str):
        w = weights.get_sharded(f"{prefix}.weight", dim=1, to_dtype=False)

        weight_scale = None
        if self.load_weight_scale:
            weight_scale = weights.get_tensor(
                f"{prefix}.weight_scale", to_dtype=False
            ).reshape(-1)

        input_scale = None
        if self.load_input_scale:
            input_scale = weights.get_tensor(
                f"{prefix}.input_scale", to_dtype=False
            ).reshape(-1)

        input_zero_point = None
        if self.load_input_zero_point:
            input_zero_point = _get_tensor_or_else(
                weights,
                f"{prefix}.input_zero_point",
                torch.zeros((1,), device=w.device, dtype=torch.int8),
            ).reshape(-1)

        return Int8Weight(
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            input_symmetric=self.input_symmetric,
            weight=w,
            weight_scale=weight_scale,
        )


OtherT = TypeVar("OtherT")


def _get_tensor_or_else(
    weights: Weights, prefix: str, other: OtherT
) -> Union[torch.Tensor, OtherT]:
    # Even if a checkpoint uses e.g. zero-points, they can be elided:
    # https://github.com/neuralmagic/compressed-tensors/blob/db6ccb25b265e8370813ecab5e95714a6728b5a6/src/compressed_tensors/compressors/quantized_compressors/base.py#L105
    if weights.has_tensor(prefix):
        return weights.get_tensor(prefix, to_dtype=False)
    else:
        return other


@dataclass
class Int8Weight(Weight):
    input_scale: Optional[torch.Tensor]
    input_zero_point: Optional[torch.Tensor]
    input_symmetric: bool
    weight: torch.Tensor
    weight_scale: Optional[torch.Tensor]

    def get_linear(self, bias: torch.Tensor):
        if self.weight_scale is None:
            assert marlin_kernels is not None
            qweight, weight_scale, _ = marlin_kernels.scaled_int8_quant(self.weight)
            return W8A8IntLinear(
                bias=bias,
                input_scale=self.input_scale,
                input_zero_point=self.input_zero_point,
                input_symmetric=self.input_symmetric,
                weight=qweight,
                weight_scale=weight_scale,
            )
        else:
            return W8A8IntLinear(
                bias=bias,
                input_scale=self.input_scale,
                input_zero_point=self.input_zero_point,
                input_symmetric=self.input_symmetric,
                weight=self.weight,
                weight_scale=self.weight_scale,
            )


class W8A8IntLinear(torch.nn.Module):
    def __init__(
        self,
        *,
        bias: Optional[torch.Tensor],
        input_scale: Optional[torch.Tensor],
        input_zero_point: Optional[torch.Tensor],
        input_symmetric: bool,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        super().__init__()

        input_scale = (
            input_scale.to(torch.float32) if input_scale is not None else input_scale
        )
        weight_scale = weight_scale.to(torch.float32)

        self.bias = bias
        self.input_symmetric = input_symmetric
        # cutlass kernels require transposed weights.
        self.weight = weight.t()
        self.weight_scale = weight_scale

        if input_scale is not None:
            if input_zero_point is None:
                # Symmetric: simply use the largest scale to cover fused layers.
                input_scale = input_scale.max()
            else:
                # Asymmetric: find the range that contains all individual ranges.
                input_zero_point = input_zero_point.to(torch.int32)
                int8_info = torch.iinfo(torch.int8)

                # Find the most extreme values of all zero point/input scale
                # pairs.
                range_min = (input_scale * (int8_info.min - input_zero_point)).min()
                range_max = (input_scale * (int8_info.max - input_zero_point)).max()

                # Calculate new scale and zero point.
                input_scale = (range_max - range_min) / (int8_info.max - int8_info.min)
                input_zero_point = int8_info.min - (range_min / input_scale)
                input_zero_point = input_zero_point.to(torch.int32)

                self.range_min = (
                    input_scale * (int8_info.min - input_zero_point)
                ).min()
                self.range_max = (
                    input_scale * (int8_info.max - input_zero_point)
                ).max()

        self.input_scale = input_scale
        self.input_zero_point = input_zero_point

        if input_symmetric:
            self.zero_point_adj = None
        else:
            # https://github.com/vllm-project/vllm/blob/8d59dbb00044a588cab96bcdc028006ed922eb06/csrc/quantization/cutlass_w8a8/Epilogues.md#scaledepilogueazp
            self.zero_point_adj = self.weight.sum(
                dim=0, keepdim=True, dtype=torch.int32
            )

            if input_zero_point is not None:
                self.zero_point_adj *= input_zero_point

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert marlin_kernels is not None

        qinput, input_scale, input_zero_point = marlin_kernels.scaled_int8_quant(
            input=input,
            scale=self.input_scale,
            azp=self.input_zero_point,
            symmetric=self.input_symmetric,
        )

        if self.input_symmetric:
            return marlin_kernels.cutlass_scaled_mm(
                a=qinput,
                b=self.weight,
                scale_a=input_scale,
                scale_b=self.weight_scale,
                out_dtype=input.dtype,
                bias=self.bias,
            )
        else:
            assert self.zero_point_adj is not None and input_scale is not None

            return marlin_kernels.cutlass_scaled_mm_azp(
                a=qinput,
                b=self.weight,
                scale_a=input_scale,
                scale_b=self.weight_scale,
                out_dtype=input.dtype,
                azp_adj=self.zero_point_adj,
                # Zero point is already in the adjustment when using static
                # input quantization.
                azp=input_zero_point if self.input_zero_point is None else None,
                bias=self.bias,
            )
