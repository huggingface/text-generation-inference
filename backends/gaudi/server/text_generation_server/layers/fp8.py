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

quant_dtype: torch.dtype = torch.float8_e4m3fn
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
if is_hpu_gaudi2():
    FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max


def pad_weight(weight, block_size):
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(
        weight, (0, pad_N, 0, pad_M), mode="constant", value=0
    )
    return padded_weight, M, N  # Return original dimensions for unpadding


def unpad_weight(weight, original_M, original_N, keep_first_dim=False):
    """Removes padding from the matrix to restore its original shape."""
    if (weight.shape[-2] == original_M) and (weight.shape[-1] == original_N):
        return weight
    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]


def pad_block_fp8_weight_naive(weight, weight_scale, block_size):

    assert len(block_size) == 2

    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n

    return weight, orig_M, orig_N


def dynamic_quant(data, single_scale=False):
    if single_scale:
        scale = ((torch.abs(data)).max() + 1e-8) / FP8_MAX
    else:
        scale = ((torch.abs(data)).max(dim=-1).values + 1e-8) / FP8_MAX
        scale = scale.unsqueeze(-1)
    data_fp8 = torch.ops.hpu.cast_to_fp8_v2(
        data, 1.0 / scale, False, False, torch.float8_e4m3fn
    )[0]
    return data_fp8, scale.float()


def dequant_block_fp8_weight_naive(
    weight,
    weight_scale,
    block_size,
    dtype=torch.bfloat16,
    original_M=None,
    original_N=None,
    do_unpad=False,
):
    if weight_scale is None:
        return weight
    assert len(block_size) == 2

    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        if is_hpu_gaudi2():
            fake_weight = weight.cpu().to(dtype).to(weight.device)
            dequant_weight = fake_weight * weight_scale.to(dtype)
        else:
            dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(
            weight_scale_m * block_size_m, weight_scale_n * block_size_n
        )
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(
            fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n
        )
        if is_hpu_gaudi2():
            fake_weight = weight.cpu().to(dtype).to(weight.device)
            dequant_weight = fake_weight * weight_scale.to(dtype)
        else:
            dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(
            fd, weight_scale_m * block_size_m, weight_scale_n * block_size_n
        )
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    if do_unpad:
        dequant_weight = unpad_weight(
            dequant_weight, original_M, original_N, keep_first_dim=keep_first_dim
        )

    return dequant_weight


def apply_block_fp8_linear_hpu_dynamic(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    x_fp8, x_scale = dynamic_quant(input_2d)

    output = torch.ops.hpu.fp8_gemm_v2(
        x_fp8,
        False,
        weight,
        True,
        None,
        torch.bfloat16,
        x_scale,
        weight_scale,
        None,
        False,
    )
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


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
    if is_hpu_gaudi2():
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

    def get_multi_weights(self, weights: "Weights", prefixes: List[str], dim: int):
        # FIXME: Force to_device to false as fp8 weights do not support torch.cat on device yet
        w = [weights.get_tensor(f"{p}.weight", to_device=False) for p in prefixes]
        shapes = [x.shape for x in w]

        # Concat then send to the device
        w = torch.cat(w, dim=dim).to(weights.device)

        # FP8 branch
        if w.dtype == torch.float8_e4m3fn:
            if self.weight_block_size is not None:
                scale = [
                    weights.get_tensor(f"{p}.weight_scale_inv", to_device=False)
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
                weights.get_tensor(f"{p}.weight_scale", to_dtype=False).reshape(-1)
                for p in prefixes
            ]
            scale = torch.cat(scale, dim=0).reshape(-1)

            input_scale = [
                weights.get_tensor(f"{p}.input_scale", to_dtype=False).reshape(-1)
                for p in prefixes
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

        if weight_block_size is not None:
            weight, orig_M, orig_N = pad_block_fp8_weight_naive(
                weight, scale, weight_block_size
            )
            weight, scale = dynamic_quant(
                dequant_block_fp8_weight_naive(
                    weight,
                    scale,
                    weight_block_size,
                    original_M=orig_M,
                    original_N=orig_N,
                    do_unpad=True,
                )
            )
            scale = scale.squeeze(-1)

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
            return apply_block_fp8_linear_hpu_dynamic(
                input, self.qweight, self.scale, self.input_scale, self.bias
            )

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
