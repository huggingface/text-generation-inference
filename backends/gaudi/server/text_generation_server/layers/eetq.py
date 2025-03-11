from dataclasses import dataclass

import torch
from EETQ import quant_weights, w8_a16_gemm
from text_generation_server.utils.weights import UnquantizedWeight


@dataclass
class EETQWeight(UnquantizedWeight):
    weight: torch.Tensor

    def get_linear(self, bias: torch.Tensor):
        try:
            from text_generation_server.layers.eetq import EETQLinear

            return EETQLinear(self.weight, bias)
        except ImportError:
            raise ImportError(
                "Please install EETQ from https://github.com/NetEase-FuXi/EETQ"
            )


class EETQLinear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        device = weight.device
        if weight.dtype != torch.float16:
            weight = weight.to(dtype=torch.float16)
        weight = torch.t(weight).contiguous().cpu()
        weight, scale = quant_weights(weight, torch.int8, False)

        self.weight = weight.cuda(device)
        self.scale = scale.cuda(device)
        self.bias = bias.cuda(device) if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = w8_a16_gemm(input, self.weight, self.scale)
        output = output + self.bias if self.bias is not None else output
        return output
