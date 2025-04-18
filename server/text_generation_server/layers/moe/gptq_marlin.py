from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from text_generation_server.layers import moe
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.kernels import load_kernel
from text_generation_server.utils.weights import Weights
from text_generation_server.layers.marlin.gptq import (
    GPTQMarlinWeight,
    GPTQMarlinWeightsLoader,
)

if SYSTEM == "cuda":
    moe_kernels = load_kernel(module="moe", repo_id="kernels-community/moe")
else:
    moe_kernels = None


try:
    major, _minor = torch.cuda.get_device_capability()
    has_sm_8_0 = major >= 8
except Exception:
    has_sm_8_0 = False


def can_use_marlin_moe_gemm(
    *,
    quant_method: str,
    quantize: str,
    sym: bool,
):
    return (
        SYSTEM == "cuda"
        and moe is not None
        and has_sm_8_0
        and quantize in {"awq", "gptq"}
        and quant_method in {"awq", "gptq"}
        # We only support asymmetric quantization for AWQ.
        and (sym or quant_method == "awq")
    )


@dataclass
class GPTQMarlinMoEWeight:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor
    perm: torch.Tensor
    is_full_k: bool


class GPTQMarlinSparseMoELayer(nn.Module):
    """
    MoE layer that uses a fused GPTQ-Marlin kernel.
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
        scoring_func: Optional[str] = None,
        e_score_correction_bias: Optional[float] = None,
    ):
        assert scoring_func in (
            "sigmoid",
            "softmax",
        ), f"scoring func {scoring_func} is not handled"
        super().__init__()

        if not (
            isinstance(weights.loader, GPTQMarlinWeightsLoader)
            and can_use_marlin_moe_gemm(
                quant_method=weights.loader.quant_method,
                quantize=weights.loader.quantize,
                sym=weights.loader.sym,
            )
        ):
            raise ValueError(
                f"Unsupported weights loader: {type(weights.loader)}, only GPTQMarlinWeightsLoader with AWQ and symmetric GPTQ quantization is supported"
            )

        assert (n_expert_group is None) == (
            topk_group is None
        ), "n_expert_group and topk_group must both be None or have some value"

        self.n_expert_group = n_expert_group
        self.topk = topk
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias

        self.gate_up_proj = _load_expert_multi_weights_col(
            prefix=prefix,
            n_experts=n_experts,
            names=[gate_proj_name, up_proj_name],
            weights=weights,
        )

        self.down_proj = _load_expert_weights_row(
            prefix=prefix, n_experts=n_experts, name=down_proj_name, weights=weights
        )

        self.bits = weights.loader.bits

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        return fused_marlin_moe(
            hidden_states=x,
            w1=self.gate_up_proj.qweight,
            w2=self.down_proj.qweight,
            w1_scale=self.gate_up_proj.scales,
            w2_scale=self.down_proj.scales,
            w1_zeros=(
                self.gate_up_proj.qzeros
                if self.gate_up_proj.qzeros.numel() > 0
                else None
            ),
            w2_zeros=(
                self.down_proj.qzeros if self.down_proj.qzeros.numel() > 0 else None
            ),
            g_idx1=self.gate_up_proj.g_idx,
            g_idx2=self.down_proj.g_idx,
            sort_indices1=self.gate_up_proj.perm,
            sort_indices2=self.down_proj.perm,
            is_k_full=self.gate_up_proj.is_full_k or self.down_proj.is_full_k,
            gating_output=gating_output,
            topk=self.topk,
            renormalize=self.renormalize,
            use_grouped_topk=self.n_expert_group is not None,
            num_expert_group=self.n_expert_group,
            topk_group=self.topk_group,
            num_bits=self.bits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
        )


def _load_expert_multi_weights_col(
    *,
    prefix: str,
    n_experts: int,
    names: List[str],
    weights: Weights,
) -> GPTQMarlinMoEWeight:
    moe_weight = None
    for i in range(n_experts):
        weight = weights.get_multi_weights_col(
            [f"{prefix}.{i}.{name}" for name in names], 0
        )
        assert isinstance(weight, GPTQMarlinWeight)
        moe_weight = _pack_weight(
            n_experts=n_experts, expert=i, weight=weight, moe_weight=moe_weight
        )
    assert moe_weight is not None
    return moe_weight


def _load_expert_weights_row(
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
) -> GPTQMarlinMoEWeight:
    moe_weight = None
    for i in range(n_experts):
        weight = weights.get_weights_row(
            f"{prefix}.{i}.{name}",
        )
        assert isinstance(weight, GPTQMarlinWeight)
        moe_weight = _pack_weight(
            n_experts=n_experts, expert=i, weight=weight, moe_weight=moe_weight
        )
    assert moe_weight is not None
    return moe_weight


def _pack_weight(
    *,
    n_experts: int,
    expert: int,
    moe_weight: Optional[GPTQMarlinMoEWeight],
    weight: GPTQMarlinWeight,
) -> GPTQMarlinMoEWeight:
    if moe_weight is None:
        qweight = torch.empty(
            (n_experts,) + weight.qweight.shape,
            dtype=weight.qweight.dtype,
            device=weight.qweight.device,
        )
        qzeros = torch.empty(
            (n_experts,) + weight.qzeros.shape,
            dtype=weight.qzeros.dtype,
            device=weight.qzeros.device,
        )
        scales = torch.empty(
            (n_experts,) + weight.scales.shape,
            dtype=weight.scales.dtype,
            device=weight.scales.device,
        )
        g_idx = torch.empty(
            (n_experts,) + weight.g_idx.shape,
            dtype=weight.g_idx.dtype,
            device=weight.g_idx.device,
        )
        perm = torch.empty(
            (n_experts,) + weight.perm.shape,
            dtype=weight.perm.dtype,
            device=weight.perm.device,
        )

        moe_weight = GPTQMarlinMoEWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            perm=perm,
            is_full_k=weight.is_full_k,
        )

    moe_weight.qweight[expert] = weight.qweight
    moe_weight.qzeros[expert] = weight.qzeros
    moe_weight.scales[expert] = weight.scales
    moe_weight.g_idx[expert] = weight.g_idx
    moe_weight.perm[expert] = weight.perm

    return moe_weight


def fused_marlin_moe(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    gating_output: torch.Tensor,
    g_idx1: torch.Tensor,
    g_idx2: torch.Tensor,
    sort_indices1: torch.Tensor,
    sort_indices2: torch.Tensor,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    is_k_full: bool,
    topk: int,
    renormalize: bool,
    num_bits: int = 8,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    topk_group: Optional[int] = None,
    scoring_func: Optional[str] = None,
    e_score_correction_bias: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (torch.Tensor): The first set of act_order indices.
    - g_idx2 (torch.Tensor): The second set of act_order indices.
    - sort_indices1 (torch.Tensor): The first act_order input permutation.
    - sort_indices2 (torch.Tensor): The second act_order input permutation.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype == torch.float16
    assert num_bits in [4, 8]

    # DeekSeekv2 uses grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        topk_weights, topk_ids = moe_kernels.grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
    elif custom_routing_function is None:
        topk_weights, topk_ids = moe_kernels.fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
    return moe_kernels.fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        gating_output=gating_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        num_bits=num_bits,
        is_k_full=is_k_full,
    )
