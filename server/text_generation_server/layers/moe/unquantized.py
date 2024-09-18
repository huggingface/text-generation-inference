from typing import Optional

import torch
import torch.nn as nn

from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.weights import UnquantizedWeight, Weights

if SYSTEM != "ipex":
    from moe_kernels.fused_moe import fused_moe


class UnquantizedSparseMoELayer(nn.Module):
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

        assert (n_expert_group is None) == (
            topk_group is None
        ), "n_expert_group and topk_group must both be None or have some value"

        self.n_expert_group = n_expert_group
        self.topk = topk
        self.topk_group = topk_group
        self.renormalize = renormalize

        self.gate_up_proj = _load_expert_multi_weights_col(
            prefix=prefix,
            n_experts=n_experts,
            gate_proj_name=gate_proj_name,
            up_proj_name=up_proj_name,
            weights=weights,
        )

        self.down_proj = _load_expert_weights_row(
            prefix=prefix,
            n_experts=n_experts,
            name=down_proj_name,
            weights=weights,
        )

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        return fused_moe(
            x,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=gating_output,
            topk=self.topk,
            renormalize=self.renormalize,
            inplace=True,
            use_grouped_topk=self.n_expert_group is not None,
            num_expert_group=self.n_expert_group,
            topk_group=self.topk_group,
        )


def _load_expert_multi_weights_col(
    *,
    prefix: str,
    n_experts: int,
    gate_proj_name: str,
    up_proj_name: str,
    weights: Weights,
) -> torch.Tensor:
    all_weight = None
    for i in range(n_experts):
        weight = weights.get_multi_weights_col(
            [f"{prefix}.{i}.{gate_proj_name}", f"{prefix}.{i}.{up_proj_name}"], 0
        )

        assert isinstance(weight, UnquantizedWeight)

        if all_weight is None:
            all_weight = torch.empty(
                (n_experts,) + weight.weight.shape,
                dtype=weight.weight.dtype,
                device=weight.weight.device,
            )

        all_weight[i] = weight.weight

    assert all_weight is not None

    return all_weight


def _load_expert_weights_row(
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
) -> torch.Tensor:
    all_weight = None
    for i in range(n_experts):
        weight = weights.get_weights_row(
            f"{prefix}.{i}.{name}",
        )

        assert isinstance(weight, UnquantizedWeight)

        if all_weight is None:
            all_weight = torch.empty(
                (n_experts,) + weight.weight.shape,
                dtype=weight.weight.dtype,
                device=weight.weight.device,
            )

        all_weight[i] = weight.weight

    assert all_weight is not None

    return all_weight
