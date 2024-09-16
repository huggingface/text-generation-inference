from typing import Optional

import torch
import torch.nn as nn
from text_generation_server.layers.fp8 import HybridFP8UnquantLoader
from text_generation_server.layers.moe.unquantized import UnquantizedSparseMoELayer
from text_generation_server.utils.weights import (
    DefaultWeightsLoader,
    UnquantizedWeight,
    Weights,
)


class SparseMoELayer(nn.Module):
    """
    Layer for MoE that uses fused kernels to only apply the active experts
    for each token (rather than applying all experts and selecting the
    outputs of active experts).
    """

    def __init__(
        self,
        *,
        n_expert_group: Optional[int],
        n_experts: int,
        prefix: str,
        renormalize: bool,
        topk: int,
        topk_group: Optional[int],
        weights: Weights,
        gate_proj_name: str = "gate_proj",
        up_proj_name: str = "up_proj",
        down_proj_name: str = "down_proj",
    ):
        super().__init__()

        if (
            isinstance(weights.loader, DefaultWeightsLoader)
            and isinstance(weights.loader.weight_class, UnquantizedWeight)
        ) or isinstance(weights.loader, HybridFP8UnquantLoader):
            cls = UnquantizedSparseMoELayer
            # Once we wire up GPTQ-Marlin MoE:
            # elif isinstance(weights.loader, GPTQMarlinWeightsLoader) and weights.loader.sym:
            # cls = GPTQMarlinSparseMoELayer
        else:
            raise ValueError(
                f"Unsupported weights loader: {weights.loader}, sparse MoE is only supported for unquantized and GPTQ weights"
            )

        self.moe = cls(
            n_expert_group=n_expert_group,
            n_experts=n_experts,
            prefix=prefix,
            renormalize=renormalize,
            topk=topk,
            topk_group=topk_group,
            weights=weights,
            gate_proj_name=gate_proj_name,
            up_proj_name=up_proj_name,
            down_proj_name=down_proj_name,
        )

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        return self.moe(x, gating_output=gating_output)

    @staticmethod
    def is_supported(weights: Weights) -> bool:
        return (
            (
                isinstance(weights.loader, DefaultWeightsLoader)
                and isinstance(weights.loader.weight_class, UnquantizedWeight)
            )
            or isinstance(weights.loader, HybridFP8UnquantLoader)
            # Once we wire up GPTQ-Marlin MoE:
            # or isinstance(weights.loader, GPTQMarlinWeightsLoader)
        )
