import torch
from typing import List, Union
from dataclasses import dataclass

from text_generation_server.utils.weights import WeightsLoader, Weights


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


class Exl2WeightsLoader(WeightsLoader):
    """Loader for exl2-quantized weights."""

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        raise RuntimeError("Column-packed weights are not supported for exl")

    def get_weights_col(self, weights: Weights, prefix: str):
        try:
            q_weight = weights.get_tensor(f"{prefix}.q_weight")
        except RuntimeError:
            raise RuntimeError(
                "Cannot load `exl2`-quantized weight, make sure the model is already quantized."
            )

        q_scale = weights.get_tensor(f"{prefix}.q_scale")
        q_invperm = weights.get_tensor(f"{prefix}.q_invperm")
        q_scale_max = weights.get_tensor(f"{prefix}.q_scale_max")
        q_groups = weights.get_tensor(f"{prefix}.q_groups")

        return Exl2Weight(
            q_weight=q_weight,
            q_scale=q_scale,
            q_invperm=q_invperm,
            q_scale_max=q_scale_max,
            q_groups=q_groups,
        )

    def get_multi_weights_col(self, weights: Weights, prefixes: List[str], dim: int):
        raise ValueError("get_multi_weights_col is not supported for exl2")

    def get_weights_row(self, weights: Weights, prefix: str):
        try:
            q_weight = weights.get_tensor(f"{prefix}.q_weight")
        except RuntimeError:
            raise RuntimeError(
                "Cannot load `exl2`-quantized weight, make sure the model is already quantized."
            )

        q_scale = weights.get_tensor(f"{prefix}.q_scale")
        q_invperm = weights.get_tensor(f"{prefix}.q_invperm")
        q_scale_max = weights.get_tensor(f"{prefix}.q_scale_max")
        q_groups = weights.get_tensor(f"{prefix}.q_groups")

        return Exl2Weight(
            q_weight=q_weight,
            q_scale=q_scale,
            q_invperm=q_invperm,
            q_scale_max=q_scale_max,
            q_groups=q_groups,
        )
