import torch
from torch.nn import functional as F


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


def get_linear(weight, bias):
    # Weights that are loaded through methods that are not
    # quantization-aware are still bare tensors. We may want
    # to change this in the future.
    if isinstance(weight, torch.Tensor):
        return FastLinear(weight, bias)

    return weight.get_linear(bias)
