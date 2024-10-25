import torch

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from loguru import logger

from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.weights import (
    Weight,
    WeightsLoader,
    UnquantizedWeight,
    Weights,
)
from text_generation_server.utils.log import log_master, log_once
import importlib.util


FBGEMM_MM_AVAILABLE = False
FBGEMM_DYN_AVAILABLE = False


def is_fbgemm_gpu_available():
    try:
        return importlib.util.find_spec("fbgemm_gpu.experimental.gen_ai") is not None
    except ModuleNotFoundError:
        return False


try:
    import marlin_kernels
except ImportError:
    marlin_kernels = None


if is_fbgemm_gpu_available():
    if SYSTEM == "cuda":
        major, _ = torch.cuda.get_device_capability()
        FBGEMM_MM_AVAILABLE = major == 9
        FBGEMM_DYN_AVAILABLE = major >= 8
else:
    log_master(logger.warning, "FBGEMM fp8 kernels are not installed.")


def get_fp8_linear() -> torch.nn.Module:
    """
    Return an FP8 linear `Module` that is compatible with the current system.
    """

    if SYSTEM == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major == 8:
            from text_generation_server.layers.marlin import GPTQMarlinFP8Linear

            return GPTQMarlinFP8Linear

    # On other systems let Torch decide if the hardware supports FP8.
    return Fp8Linear


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale


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
    if FBGEMM_DYN_AVAILABLE and not scalar and not scale:
        qweight, scale = torch.ops.fbgemm.quantize_fp8_per_row(
            weight, bs=None, scale_ub=scale_upper_bound, output_dtype=qdtype
        )
        return qweight, scale

    if marlin_kernels is not None:
        shape = weight.shape
        qweight, scale = marlin_kernels.scaled_fp8_quant(
            weight.reshape(-1, shape[-1]),
            dtype=qdtype,
            scale=scale,
            scale_ub=scale_upper_bound,
        )

        return qweight.reshape(shape), scale

    # weight, scale = quant_weights(weight, torch.int8, False)
    finfo = torch.finfo(qdtype)

    if scale is None:
        # Calculate the scale as dtype max divided by absmax
        scale = finfo.max / weight.abs().max().clamp(min=1e-12, max=scale_upper_bound)
        # scale and clamp the tensor to bring it to
        # the representative range of float8 data type
        # (as default cast is unsaturated)
        qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
        scale = scale.float().reciprocal()
    else:
        # Use reciprocal to avoid more expensive division.
        qweight = (weight * scale.reciprocal()).clamp(min=finfo.min, max=finfo.max)

    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)

    if SYSTEM == "rocm":
        qweight, scale, _ = normalize_e4m3fn_to_e4m3fnuz(qweight, scale)

    return qweight, scale


class HybridFP8UnquantLoader(WeightsLoader):
    """Weight loader that loads FP8 and unquantized Torch tensors."""

    def __init__(self, activation_scale_ub: Optional[float], to_fp8: bool):
        self.activation_scale_ub = activation_scale_ub
        self.to_fp8 = to_fp8

    def get_weights(self, weights: "Weights", prefix: str):
        w = weights.get_tensor(f"{prefix}.weight")

        if w.dtype == torch.float8_e4m3fn:
            # FP8 branch
            scale = (
                weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)
                .reshape(-1)
                .expand(w.shape[0])
            )

            input_scale = None
            if weights.has_tensor(f"{prefix}.input_scale"):
                input_scale = weights.get_tensor(
                    f"{prefix}.input_scale", to_dtype=False
                ).reshape(-1)

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
            scale = scale.reshape(-1).expand(w.shape[0])

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
            scale = (
                weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)
                .reshape(-1)
                .expand(w.shape[0])
            )
            input_scale = None
            if weights.has_tensor(f"{prefix}.input_scale"):
                input_scale = weights.get_tensor(
                    f"{prefix}.input_scale", to_dtype=False
                ).reshape(-1)

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

    def get_linear(self, bias: torch.Tensor):
        if self.weight_scale is None:
            return get_fp8_linear().from_unquant(self.weight, bias, self.dtype)
        # This is not checked by the fbgemm kernels, but they require contiguous
        # memory. Can be non-contiguous when we e.g. expand from scalars.
        self.weight_scale = self.weight_scale.contiguous()
        return get_fp8_linear().from_fp8(
            weight=self.weight,
            scale=self.weight_scale,
            dtype=self.dtype,
            bias=bias,
            input_scale=self.input_scale,
            scale_upper_bound=self.activation_scale_ub,
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
    ) -> None:
        super().__init__()
        if FBGEMM_MM_AVAILABLE:
            log_once(logger.info, "Using FBGEMM fp8 optimized kernels")
        if SYSTEM == "rocm" and qweight.dtype == torch.float8_e4m3fn:
            qweight, scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=qweight, weight_scale=scale
            )

        self.dtype = dtype
        self.qweight = qweight
        self.scale = scale.float()
        self.input_scale = input_scale.float() if input_scale is not None else None

        if FBGEMM_MM_AVAILABLE:
            self.scale_upper_bound = (
                torch.tensor(
                    [scale_upper_bound], dtype=torch.float32, device=qweight.device
                )
                if scale_upper_bound is not None
                else None
            )
        else:
            self.scale_upper_bound = scale_upper_bound

        self.bias = bias if bias is not None else None

    @classmethod
    def from_unquant(cls, weight, bias, dtype):
        qweight, scale = fp8_quantize(weight, scalar=not FBGEMM_MM_AVAILABLE)
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

        if FBGEMM_DYN_AVAILABLE:
            # fbgemm needs float32 scales.
            scale = scale.float()
        return cls(
            qweight=weight,
            scale=scale,
            input_scale=input_scale,
            scale_upper_bound=scale_upper_bound,
            bias=bias,
            dtype=dtype,
        )

    @classmethod
    def get_shared_device_identity(cls, device):
        # Input scaling factors are no longer optional in _scaled_mm starting
        # from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
        if device not in cls._device_identity_cache:
            cls._device_identity_cache[device] = torch.ones(1, device=device)
        return cls._device_identity_cache[device]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if FBGEMM_MM_AVAILABLE:
            qinput, scale = fp8_quantize(
                input, scale_upper_bound=self.scale_upper_bound
            )

            y = torch.ops.fbgemm.f8f8bf16_rowwise(
                qinput,
                self.qweight,
                scale,
                self.scale,
                use_fast_accum=True,
                bias=self.bias,
            )
            return y.to(self.dtype)

        qinput, scale = fp8_quantize(
            input,
            self.input_scale,
            scale_upper_bound=self.scale_upper_bound,
            scalar=True,
        )

        per_tensor_weights = self.scale.numel() == 1
        per_tensor_activations = scale.numel() == 1

        if SYSTEM != "rocm" or (per_tensor_weights and per_tensor_activations):
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
        else:
            device_identity = None
            if SYSTEM == "rocm":
                device_identity = self.get_shared_device_identity(self.qweight.device)

            output = torch._scaled_mm(
                qinput,
                self.qweight.t(),
                scale_a=device_identity,
                scale_b=device_identity,
                out_dtype=torch.float32,
            )
            if isinstance(output, tuple) and len(output) == 2:
                output = output[0]

            output = output * scale * self.scale.t()
            if self.bias is not None:
                output = output + self.bias

            output = output.to(dtype=self.dtype)

        return output


def _load_scalar_or_matrix_scale(weights: Weights, prefix: str, shape: torch.Size):
    scale = weights.get_tensor(prefix, to_dtype=False)
    if scale.numel() > 1:
        scale = weights.get_sharded(prefix, dim=0, to_dtype=False)
    return scale.reshape(-1).expand(shape[0])
