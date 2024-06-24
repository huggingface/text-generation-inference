import torch
from dataclasses import dataclass


@dataclass
class Exl2Weight:
    """
    Exllama2 exl2 quantized weights.
    """

    q_weight: torch.Tensor
    q_scale: torch.Tensor
    q_invperm: torch.Tensor
    q_scale_max: torch.Tensor
    q_groups: torch.Tensor

    def __post_init__(self):
        self.q_scale_max /= 256
        self.q_invperm = self.q_invperm.short()

    @property
    def device(self) -> torch.device:
        return self.q_weight.device
