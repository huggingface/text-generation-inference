import torch

from dataclasses import dataclass

from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.weights import Weight

try:
    import fbgemm_gpu.experimental.gen_ai

    HAS_FBGEMM = True
except (ImportError, ModuleNotFoundError):
    HAS_FBGEMM = False


def get_fp8_linear() -> torch.nn.Module:
    """
    Return an FP8 linear `Module` that is compatible with the current system.
    """

    if SYSTEM == "cuda":
        major, minor = torch.cuda.get_device_capability()
        if major == 8 and minor < 9:
            from text_generation_server.layers.marlin import GPTQMarlinFP8Linear

            return GPTQMarlinFP8Linear

    # On other systems let Torch decide if the hardware supports FP8.
    return Fp8Linear


def fp8_quantize(weight, scale_upper_bound=None, qdtype=torch.float8_e4m3fn):
    if HAS_FBGEMM:
        if scale_upper_bound.device != weight.device:
            scale_upper_bound = scale_upper_bound.to(weight.device)

        qweight, scale = torch.ops.fbgemm.quantize_fp8_per_row(
            weight, bs=None, scale_ub=scale_upper_bound, output_dtype=qdtype
        )
        return qweight, scale

    # weight, scale = quant_weights(weight, torch.int8, False)
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12, max=scale_upper_bound)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale


@dataclass
class Fp8Weight(Weight):
    weight: torch.Tensor

    def get_linear(self, bias: torch.Tensor):
        return get_fp8_linear()(self.weight, bias)


class Fp8Linear(torch.nn.Module):
    def __init__(
        self,
        qweight,
        scale,
        scale_upper_bound,
        bias,
        dtype,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.qweight = qweight
        self.scale = scale
        self.scale_upper_bound = scale_upper_bound

        self.bias = bias if bias is not None else None

    @classmethod
    def from_unquant(cls, weight, bias, dtype):
        qweight, scale = fp8_quantize(weight)
        return cls(
            qweight=qweight, scale=scale, scale_upper_bound=None, bias=bias, dtype=dtype
        )

    @classmethod
    def from_fp8(cls, weight, bias, dtype):
        return cls(
            qweight=weight.weight,
            scale=weight.weight_scale,
            scale_upper_bound=weight.input_scale,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if HAS_FBGEMM:
            qinput, scale = fp8_quantize(
                input, scale_upper_bound=self.scale_upper_bound
            )

            y = torch.ops.fbgemm.f8f8bf16_rowwise(
                qinput,
                self.weight,
                scale,
                self.scale,
                use_fast_accum=True,
                bias=self.bias,
            )
            return y.to(self.dtype)

        qinput, scale = fp8_quantize(input)
        output, _ = torch._scaled_mm(
            qinput,
            self.qweight.t(),
            out_dtype=self.dtype,
            scale_a=scale,
            scale_b=self.scale,
            bias=self.bias,
        )
        return output
