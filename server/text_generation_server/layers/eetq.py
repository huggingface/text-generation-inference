import torch
from EETQ import quant_weights, w8_a16_gemm


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
