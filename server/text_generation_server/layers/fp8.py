from dataclasses import dataclass

import torch
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.weights import Weight


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


def fp8_quantize(weight, qdtype=torch.float8_e4m3fn):
    device = weight.device
    # weight, scale = quant_weights(weight, torch.int8, False)
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12)
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
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.dtype = weight.dtype
        self.qweight, self.scale = fp8_quantize(weight)

        self.bias = bias if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
