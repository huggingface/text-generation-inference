import torch
from torch import nn


class FastLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(FastLinear, self).__init__(in_features, out_features, bias, device, dtype)

    def transpose_weight(self):
        self.weight = nn.Parameter(self.weight.T)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.addmm(self.bias, input, self.weight)
        return torch.matmul(input, self.weight)
