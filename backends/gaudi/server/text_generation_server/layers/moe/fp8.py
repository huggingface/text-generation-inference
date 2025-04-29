from typing import Optional

import torch
import torch.nn as nn

from text_generation_server.utils.weights import Weights
from text_generation_server.layers.fp8 import (
    Fp8Weight,
    fp8_quantize,
    quant_dtype,
    normalize_e4m3fn_to_native_float8,
    dynamic_quant,
    dequant_block_fp8_weight_naive,
)
from text_generation_server.layers.moe.fused_moe import select_experts
import habana_frameworks.torch as htorch


class FP8SparseMoELayer(nn.Module):
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

        (
            self.gate_up_proj,
            self.gate_up_proj_weight_scale,
            self.gate_up_proj_input_scale,
        ) = _load_expert_multi_weights_col(
            prefix=prefix,
            n_experts=n_experts,
            gate_proj_name=gate_proj_name,
            up_proj_name=up_proj_name,
            weights=weights,
        )

        self.down_proj, self.down_proj_weight_scale, self.down_proj_input_scale = (
            _load_expert_weights_row(
                prefix=prefix,
                n_experts=n_experts,
                name=down_proj_name,
                weights=weights,
            )
        )
        if self.weight_block_size is not None:
            self.gate_up_proj, self.gate_up_proj_weight_scale = dynamic_quant(
                dequant_block_fp8_weight_naive(
                    self.gate_up_proj,
                    self.gate_up_proj_weight_scale,
                    self.weight_block_size,
                )
            )
            self.down_proj, self.down_proj_weight_scale = dynamic_quant(
                dequant_block_fp8_weight_naive(
                    self.down_proj, self.down_proj_weight_scale, self.weight_block_size
                )
            )
            self.gate_up_proj_weight_scale, self.down_proj_weight_scale = (
                self.gate_up_proj_weight_scale.squeeze(-1),
                self.down_proj_weight_scale.squeeze(-1),
            )

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=gating_output,
            use_grouped_topk=self.n_expert_group is not None,
            top_k=self.topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.n_expert_group,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
        )
        total_num_experts = gating_output.size(-1)
        x_fp8, x_scale = dynamic_quant(x, single_scale=True)
        moe_n_slice = (total_num_experts + 31) // 32
        n_expert_slice = (total_num_experts + moe_n_slice - 1) // moe_n_slice
        for i in range(moe_n_slice):
            min_expert = i * n_expert_slice
            max_expert = min((i + 1) * n_expert_slice, total_num_experts)
            w13_list_slice = [
                self.gate_up_proj[j, ...] for j in range(min_expert, max_expert)
            ]
            w2_list_slice = [
                self.down_proj[j, ...] for j in range(min_expert, max_expert)
            ]
            w13_weight_scale = [
                self.gate_up_proj_weight_scale[j, ...]
                for j in range(min_expert, max_expert)
            ]
            w2_weight_scale = [
                self.down_proj_weight_scale[j, ...]
                for j in range(min_expert, max_expert)
            ]

            current_hidden_states = torch.ops.hpu.mixture_of_experts(
                hidden_states=x_fp8,
                expert_routing_table=topk_ids.to(torch.int64),
                router_weights=topk_weights.to(x.dtype),
                w12=w13_list_slice,
                w3=w2_list_slice,
                d_scale_hidden_states=x_scale,
                d_scale_w12=w13_weight_scale,
                d_scale_w3=w2_weight_scale,
                permuted_weights=True,
                activation="silu",
                experts_min=min_expert,
                experts_max=max_expert - 1,
            )
            htorch.core.mark_step()
            if i == 0:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)
        return final_hidden_states


def _load_expert_weights(
    get_weight_fn,
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
) -> torch.Tensor:
    all_weight = None
    all_weight_scales = None
    max_input_scale = None

    for i in range(n_experts):
        weight = get_weight_fn(prefix, i, name, weights)

        assert isinstance(weight, Fp8Weight)

        if all_weight is None:
            all_weight = torch.empty(
                (n_experts,) + weight.weight.shape,
                dtype=quant_dtype,
                device=weight.weight.device,
            )
        if all_weight_scales is None:
            all_weight_scales = torch.empty(
                (n_experts,) + weight.weight_scale.shape,
                dtype=torch.float32,
                device=weight.weight.device,
            )

        if weight.weight.dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
            all_weight[i], all_weight_scales[i], current_input_scale = (
                normalize_e4m3fn_to_native_float8(
                    weight.weight, weight.weight_scale, weight.input_scale
                )
            )
            if current_input_scale is not None:
                if max_input_scale is None or current_input_scale > max_input_scale:
                    max_input_scale = current_input_scale
        else:
            all_weight[i], all_weight_scales[i] = fp8_quantize(
                weight.weight, scalar=True
            )

    assert all_weight is not None

    return all_weight, all_weight_scales, max_input_scale


def _load_expert_multi_weights_col(
    *,
    prefix: str,
    n_experts: int,
    gate_proj_name: str,
    up_proj_name: str,
    weights: Weights,
) -> torch.Tensor:
    def get_weight_fn(prefix, i, name, weights):
        return weights.get_multi_weights_col(
            [f"{prefix}.{i}.{gate_proj_name}", f"{prefix}.{i}.{up_proj_name}"], 0
        )

    return _load_expert_weights(
        get_weight_fn, prefix=prefix, n_experts=n_experts, name=None, weights=weights
    )


def _load_expert_weights_row(
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
) -> torch.Tensor:
    def get_weight_fn(prefix, i, name, weights):
        return weights.get_weights_row(f"{prefix}.{i}.{name}")

    return _load_expert_weights(
        get_weight_fn, prefix=prefix, n_experts=n_experts, name=name, weights=weights
    )
