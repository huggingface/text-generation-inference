from typing import Optional

import torch
import torch.nn as nn

from text_generation_server.utils.weights import UnquantizedWeight, Weights
from vllm_hpu_extension.ops import VllmMixtureOfExpertsOp
import habana_frameworks.torch as htorch
import torch.nn.functional as F
import os


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
        scoring_func: Optional[str] = "softmax",
        e_score_correction_bias: Optional[float] = None,
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
        self.weight_block_size = weights.weights_loader.weight_block_size
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.rank = weights.process_group.rank()
        self.world_size = weights.process_group.size()
        self.use_ep = os.getenv("USE_EXPERT_PARALLEL", "true").lower() == "true"
        if (n_experts + self.world_size - 1) // self.world_size < 4:
            self.use_ep = False
        if self.use_ep:
            n_experts_per_rank = (n_experts + self.world_size - 1) // self.world_size
            self.ep_offset = self.rank * n_experts_per_rank
            n_experts = min(n_experts_per_rank, n_experts - self.ep_offset)
            experts_min = self.ep_offset
            experts_max = self.ep_offset + n_experts - 1
        else:
            self.ep_offset = 0
            experts_min = 0
            experts_max = n_experts - 1

        self.gate_up_proj = _load_expert_multi_weights_col(
            prefix=prefix,
            n_experts=n_experts,
            gate_proj_name=gate_proj_name,
            up_proj_name=up_proj_name,
            weights=weights,
            use_ep=self.use_ep,
            ep_offset=self.ep_offset,
        )

        self.down_proj = _load_expert_weights_row(
            prefix=prefix,
            n_experts=n_experts,
            name=down_proj_name,
            weights=weights,
            use_ep=self.use_ep,
            ep_offset=self.ep_offset,
        )

        self.MoeOp = VllmMixtureOfExpertsOp(n_experts, experts_min, experts_max)
        for i in range(n_experts):
            self.MoeOp.w13_list[i].set_weight(self.gate_up_proj[i])
            self.MoeOp.w2_list[i].set_weight(self.down_proj[i])

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        htorch.core.mark_step()
        routing_weights = F.softmax(gating_output, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = self.MoeOp(
            hidden_states=x,
            expert_routing_table=selected_experts,
            router_weights=routing_weights,
            permuted_weights=True,
            activation="silu",
        )

        return final_hidden_states.view(-1, x.shape[1])


def _load_expert_multi_weights_col(
    *,
    prefix: str,
    n_experts: int,
    gate_proj_name: str,
    up_proj_name: str,
    weights: Weights,
    use_ep: bool = False,
    ep_offset: int = 0,
) -> torch.Tensor:
    all_weight = None
    for i in range(n_experts):
        if not use_ep:
            weight = weights.get_multi_weights_col(
                [f"{prefix}.{i}.{gate_proj_name}", f"{prefix}.{i}.{up_proj_name}"], 0
            )
        else:
            weight = weights.get_multi_weights(
                [
                    f"{prefix}.{i+ep_offset}.{gate_proj_name}",
                    f"{prefix}.{i+ep_offset}.{up_proj_name}",
                ],
                0,
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
    use_ep: bool = False,
    ep_offset: int = 0,
) -> torch.Tensor:
    all_weight = None
    for i in range(n_experts):
        if not use_ep:
            weight = weights.get_weights_row(
                f"{prefix}.{i}.{name}",
            )
        else:
            weight = weights.get_weights(
                f"{prefix}.{i+ep_offset}.{name}",
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
