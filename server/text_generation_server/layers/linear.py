from typing import Optional

import torch
from text_generation_server.utils.import_utils import SYSTEM
from torch.nn import functional as F

if SYSTEM == "rocm":
    try:
        from vllm import _custom_C
    except Exception as e:
        raise ImportError(f"Could not load `vllm._custom_C`. Full error: {e}")


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

        if SYSTEM == "rocm" and inp.numel() // inp.shape[-1] == 1:
            batched = False
            inp_shape = inp.shape

            if inp.dim() == 3:
                inp = inp.view(-1, inp_shape[-1])
                batched = True

            m, k = weight.shape[0], inp_shape[1]
            out = torch.empty(
                inp_shape[0], weight.shape[0], dtype=inp.dtype, device="cuda"
            )
            if (k == 8192 and (m == 1280 or m == 7168)) or (k == 3584 and m == 8192):
                _custom_C.LLMM1(weight, inp, out, 8)
            elif k <= 8192 and k % 8 == 0 and m % 4 == 0:
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
