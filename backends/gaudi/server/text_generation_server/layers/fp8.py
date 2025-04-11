from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union, List

import torch

from text_generation_server.utils.weights import (
    Weight,
    WeightsLoader,
    UnquantizedWeight,
    Weights,
)

from vllm_hpu_extension.ops import scaled_fp8_quant
from vllm_hpu_extension.scales import get_hpu_gaudi2_scale_factor, is_hpu_gaudi2
import habana_frameworks.torch.utils.experimental as htexp

w8a8_block_fp8_matmul = None
per_token_group_quant_fp8 = None
quant_dtype: torch.dtype = torch.float8_e4m3fn


def get_fp8_linear(force_w8a16: bool = False) -> Type[torch.nn.Module]:
    """
    Return an FP8 linear `Module` that is compatible with the current system.
    """
    # On other systems let Torch decide if the hardware supports FP8.
    return Fp8Linear


def normalize_e4m3fn_to_native_float8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    return weight, weight_scale, input_scale


def per_tensor_dequantize(
    tensor: torch.Tensor,
    inv_scale: Union[float, torch.Tensor],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    device = tensor.device
    dtype = torch.bfloat16
    if htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2:
        # dequant on cpu to avoid nan on gaudi2
        tensor = tensor.to("cpu")

    fake_qweight = tensor.to(dtype).to(device)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def requantize_with_max_scale(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    if is_hpu_gaudi2():
        max_w_scale = max_w_scale * get_hpu_gaudi2_scale_factor()

    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_dq = per_tensor_dequantize(
            weight[start:end, :], weight_scale[idx], dtype
        )
        weight[start:end, :], max_w_scale_normalized = fp8_quantize(
            weight_dq, max_w_scale
        )
        start = end

    return weight, max_w_scale_normalized


def fp8_quantize(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    scale_upper_bound: Optional[torch.Tensor] = None,
    qdtype: torch.dtype = torch.float8_e4m3fn,
    scalar: bool = False,
):
    """
    This function returns a reciprocal of the scale, so that a tensor can be unscaled
    by multiplying it with the returned scale. If a scale is given through the `scale`
    argument, it must also be a reciprocal (so that scales from an FP8 checkpoint can
    be used without modification).
    """
    shape = weight.shape
    qweight, scale = scaled_fp8_quant(
        weight.reshape(-1, shape[-1]),
        scale=scale,
        scale_ub=scale_upper_bound,
        # TODO: don't do this when we have to use the Torch kernel.
        use_per_token_if_dynamic=not scalar,
    )

    return qweight.reshape(shape), scale


class HybridFP8UnquantLoader(WeightsLoader):
    """Weight loader that loads FP8 and unquantized Torch tensors."""

    def __init__(
        self,
        activation_scale_ub: Optional[float],
        to_fp8: bool,
        weight_block_size: Optional[List[int]] = None,
    ):
        self.activation_scale_ub = activation_scale_ub
        self.to_fp8 = to_fp8
        self.weight_block_size = weight_block_size

    def get_weights(self, weights: "Weights", prefix: str):
        w = weights.get_tensor(f"{prefix}.weight")

        if w.dtype == torch.float8_e4m3fn:
            if self.weight_block_size is not None:
                scale = weights.get_tensor(f"{prefix}.weight_scale_inv")
                return Fp8Weight(
                    weight=w,
                    weight_scale=scale,
                    activation_scale_ub=self.activation_scale_ub,
                    dtype=weights.dtype,
                    weight_block_size=self.weight_block_size,
                )
            # FP8 branch
            scale = weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)

            input_scale = None
            if weights.has_tensor(f"{prefix}.input_scale"):
                input_scale = (
                    weights.get_tensor(f"{prefix}.input_scale", to_dtype=False)
                    .reshape(-1)
                    .max()
                )
            logical_widths = [w.shape[0]]
            w, scale = requantize_with_max_scale(
                w, scale.unsqueeze(0), logical_widths, weights.dtype
            )

            return Fp8Weight(
                weight=w,
                weight_scale=scale,
                input_scale=input_scale,
                activation_scale_ub=self.activation_scale_ub,
                dtype=weights.dtype,
            )
        if self.to_fp8:
            return Fp8Weight(weight=w, dtype=weights.dtype)

        return UnquantizedWeight(w)

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        w = weights.get_packed_sharded(
            f"{prefix}.weight", dim=0, block_sizes=block_sizes
        )

        if w.dtype == torch.float8_e4m3fn:
            # FP8 branch
            scale = weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)

            if scale.numel() > 1:
                scale = weights.get_packed_sharded(
                    f"{prefix}.weight_scale",
                    dim=0,
                    block_sizes=block_sizes,
                    to_dtype=False,
                )

            input_scale = None
            if weights.has_tensor(f"{prefix}.input_scale"):
                input_scale = weights.get_tensor(
                    f"{prefix}.input_scale", to_dtype=False
                )
                if input_scale.numel() > 1:
                    input_scale = weights.get_packed_sharded(
                        f"{prefix}.input_scale",
                        dim=0,
                        block_sizes=block_sizes,
                        to_dtype=False,
                    )
                input_scale = input_scale.reshape(-1).max()
            logical_widths = [w.shape[0]]
            w, scale = requantize_with_max_scale(
                w, scale.unsqueeze(0), logical_widths, weights.dtype
            )

            return Fp8Weight(
                weight=w,
                weight_scale=scale,
                input_scale=input_scale,
                activation_scale_ub=self.activation_scale_ub,
                dtype=weights.dtype,
            )
        if self.to_fp8:
            return Fp8Weight(weight=w, dtype=weights.dtype)

        return UnquantizedWeight(w)

    def get_multi_weights_col(self, weights: "Weights", prefixes: List[str], dim: int):
        # FIXME: Force to_device to false as fp8 weights do not support torch.cat on device yet
        w = [
            weights.get_sharded(f"{p}.weight", dim=0, to_device=False) for p in prefixes
        ]
        shapes = [x.shape for x in w]

        # Concat then send to the device
        w = torch.cat(w, dim=dim).to(weights.device)

        # FP8 branch
        if w.dtype == torch.float8_e4m3fn:
            if self.weight_block_size is not None:
                scale = [
                    weights.get_sharded(f"{p}.weight_scale_inv", dim=0, to_device=False)
                    for p in prefixes
                ]
                scale = torch.cat(scale, dim=dim)
                scale = scale.to(weights.device)
                return Fp8Weight(
                    weight=w,
                    weight_scale=scale,
                    activation_scale_ub=self.activation_scale_ub,
                    dtype=weights.dtype,
                    weight_block_size=self.weight_block_size,
                )

            scale = [
                _load_scalar_or_matrix_scale(weights, f"{p}.weight_scale", shape)
                for p, shape in zip(prefixes, shapes)
            ]
            scale = torch.cat(scale, dim=0).reshape(-1)

            input_scale = [
                _load_scalar_or_matrix_scale(weights, f"{p}.input_scale", shape)
                for p, shape in zip(prefixes, shapes)
                if weights.has_tensor(f"{p}.input_scale")
            ]
            assert len(input_scale) == 0 or len(input_scale) == len(prefixes)
            input_scale = (
                torch.cat(input_scale, dim=0).reshape(-1).max()
                if len(input_scale) != 0
                else None
            )

            logical_widths = [x[0] for x in shapes]
            w, scale = requantize_with_max_scale(
                w, scale.to(weights.device), logical_widths, weights.dtype
            )

            return Fp8Weight(
                weight=w,
                weight_scale=scale,
                input_scale=input_scale,
                activation_scale_ub=self.activation_scale_ub,
                dtype=weights.dtype,
            )
        if self.to_fp8:
            return Fp8Weight(weight=w, dtype=weights.dtype)

        return UnquantizedWeight(w)

    def get_weights_row(self, weights: "Weights", prefix: str):
        w = weights.get_sharded(f"{prefix}.weight", dim=1)
        # FP8 branch
        if w.dtype == torch.float8_e4m3fn:
            if self.weight_block_size is not None:
                # XXX: Yes the weights is named scale_inv, but corresponds to scale it seems.
                scale = weights.get_sharded(f"{prefix}.weight_scale_inv", dim=1)

                return Fp8Weight(
                    weight=w,
                    weight_scale=scale,
                    activation_scale_ub=self.activation_scale_ub,
                    dtype=weights.dtype,
                    weight_block_size=self.weight_block_size,
                )

            scale = weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)

            input_scale = None
            if weights.has_tensor(f"{prefix}.input_scale"):
                input_scale = (
                    weights.get_tensor(f"{prefix}.input_scale", to_dtype=False)
                    .reshape(-1)
                    .max()
                )
            logical_widths = [w.shape[0]]
            w, scale = requantize_with_max_scale(
                w, scale.unsqueeze(0), logical_widths, weights.dtype
            )
            return Fp8Weight(
                weight=w,
                weight_scale=scale,
                input_scale=input_scale,
                activation_scale_ub=self.activation_scale_ub,
                dtype=weights.dtype,
            )
        if self.to_fp8:
            return Fp8Weight(weight=w, dtype=weights.dtype)

        return UnquantizedWeight(w)


@dataclass
class Fp8Weight(Weight):
    weight: torch.Tensor
    dtype: torch.dtype
    weight_scale: Optional[torch.Tensor] = None
    input_scale: Optional[torch.Tensor] = None
    activation_scale_ub: Optional[float] = None
    force_w8a16: bool = False
    weight_block_size: Optional[List[int]] = None

    def get_linear(self, bias: torch.Tensor):
        if self.weight_scale is None:
            return get_fp8_linear(force_w8a16=self.force_w8a16).from_unquant(
                self.weight, bias, self.dtype
            )
        # This is not checked by the fbgemm kernels, but they require contiguous
        # memory. Can be non-contiguous when we e.g. expand from scalars.
        self.weight_scale = self.weight_scale.contiguous()
        return get_fp8_linear(force_w8a16=self.force_w8a16).from_fp8(
            weight=self.weight,
            scale=self.weight_scale,
            dtype=self.dtype,
            bias=bias,
            input_scale=self.input_scale,
            scale_upper_bound=self.activation_scale_ub,
            weight_block_size=self.weight_block_size,
        )


class Fp8Linear(torch.nn.Module):
    _device_identity_cache = {}

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        scale_upper_bound: Optional[float] = None,
        weight_block_size: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.dtype = dtype
        self.qweight = qweight
        self.scale = scale.float()
        self.input_scale = input_scale.float() if input_scale is not None else None
        self.weight_block_size = weight_block_size
        self.scale_upper_bound = scale_upper_bound

        self.bias = bias if bias is not None else None

    @classmethod
    def from_unquant(cls, weight, bias, dtype):
        qweight, scale = fp8_quantize(weight, scalar=True)
        return cls(
            qweight=qweight,
            scale=scale,
            dtype=dtype,
            bias=bias,
            input_scale=None,
            scale_upper_bound=None,
        )

    @classmethod
    def from_fp8(
        cls,
        weight: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "Fp8Linear":
        input_scale = kwargs.get("input_scale", None)
        scale_upper_bound = kwargs.get("scale_upper_bound", None)
        weight_block_size = kwargs.get("weight_block_size", None)

        return cls(
            qweight=weight,
            scale=scale,
            input_scale=input_scale,
            scale_upper_bound=scale_upper_bound,
            bias=bias,
            dtype=dtype,
            weight_block_size=weight_block_size,
        )

    @classmethod
    def get_shared_device_identity(cls, device):
        # Input scaling factors are no longer optional in _scaled_mm starting
        # from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
        if device not in cls._device_identity_cache:
            cls._device_identity_cache[device] = torch.ones(1, device=device)
        return cls._device_identity_cache[device]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_block_size is not None:
            # https://arxiv.org/pdf/2412.19437
            # At a more granular level. As illustrated in Figure 7 (a), (1) for activations, we group and
            # scale elements on a 1x128 tile basis (i.e., per token per 128 channels); and (2) for weights, we
            # group and scale elements on a 128x128 block basis (i.e., per 128 input channels per 128 output
            # channels).
            qinput, scale = per_token_group_quant_fp8(input, self.weight_block_size[1])
            output = w8a8_block_fp8_matmul(
                qinput,
                self.qweight,
                scale,
                self.scale,
                self.weight_block_size,
                output_dtype=input.dtype,
            )

            if self.bias is not None:
                output = output + self.bias
            return output.to(dtype=input.dtype)

        qinput, scale = fp8_quantize(
            input,
            self.input_scale,
            scale_upper_bound=self.scale_upper_bound,
            scalar=True,
        )

        output = torch._scaled_mm(
            qinput,
            self.qweight.t(),
            out_dtype=self.dtype,
            scale_a=scale,
            scale_b=self.scale,
            bias=self.bias,
        )

        if isinstance(output, tuple) and len(output) == 2:
            output = output[0]

        return output


def _load_scalar_or_matrix_scale(weights: Weights, prefix: str, shape: torch.Size):
    scale = weights.get_tensor(prefix, to_dtype=False)

    if scale.numel() > 1:
        scale = weights.get_sharded(prefix, dim=0, to_dtype=False)
    return scale.reshape(-1)
