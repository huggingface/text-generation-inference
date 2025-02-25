import torch
from text_generation_server.utils.import_utils import SYSTEM
from torch.nn import functional as F
import os

if SYSTEM == "rocm":
    ROCM_USE_SKINNY_GEMM = os.getenv("ROCM_USE_SKINNY_GEMM", "True").lower() in (
        "true",
        "1",
    )

    if ROCM_USE_SKINNY_GEMM:
        try:
            from vllm import _custom_C
        except Exception as e:
            raise ImportError(
                f"Could not load `vllm._custom_C` for ROCm skinny gemm. Full error: {e}"
            )


class FastLinear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class FastLinearROCm(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

        self.cu_count = torch.cuda.get_device_properties(
            device="cuda"
        ).multi_processor_count
        self.use_skinny_gemm = (
            ROCM_USE_SKINNY_GEMM
            and "gfx1" not in torch.cuda.get_device_properties("cuda").gcnArchName
        )

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if (
            self.use_skinny_gemm
            and inp.dtype == torch.float16
            and inp.shape[-1] % 8 == 0
        ):
            batched = False
            inp_shape = inp.shape

            if inp.dim() == 3:
                inp = inp.view(-1, inp_shape[-1])
                batched = True

            m, n, k = weight.shape[0], inp_shape[0], inp_shape[1]
            if m > 8 and n <= 4:
                out = torch.empty(
                    inp_shape[0], weight.shape[0], dtype=inp.dtype, device=weight.device
                )
                _custom_C.wvSpltK(weight, inp, out, n, self.cu_count)
            elif m % 4 == 0 and n == 1 and k <= 8192:
                out = torch.empty(
                    inp_shape[0], weight.shape[0], dtype=inp.dtype, device=weight.device
                )
                _custom_C.LLMM1(weight, inp, out, 4)
            else:
                out = F.linear(inp, weight)

            if batched:
                out.view(*inp_shape[:-1], out.shape[-1])

            if bias is not None:
                out = out + bias
            return out
        return F.linear(inp, self.weight, self.bias)


def get_linear(weight, bias):
    # Weights that are loaded through methods that are not
    # quantization-aware are still bare tensors. We may want
    # to change this in the future.
    if isinstance(weight, torch.Tensor):
        if SYSTEM == "rocm":
            return FastLinearROCm(weight, bias)
        else:
            return FastLinear(weight, bias)

    return weight.get_linear(bias)
