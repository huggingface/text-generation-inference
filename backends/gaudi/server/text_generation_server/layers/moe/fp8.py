from typing import Optional

import torch
import torch.nn as nn

from text_generation_server.utils.weights import Weights
from text_generation_server.layers.fp8 import (
    Fp8Weight,
    fp8_quantize,
    quant_dtype,
    normalize_e4m3fn_to_native_float8,
)

try:
    from .unquantized import fused_moe
except Exception:
    fused_moe = None


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
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            use_fp8_w8a8=True,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
            a1_scale=self.gate_up_proj_input_scale,
            a2_scale=self.down_proj_input_scale,
        )


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
