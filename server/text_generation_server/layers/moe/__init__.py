from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from loguru import logger
from transformers.activations import ACT2FN

from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)
from text_generation_server.layers.fp8 import HybridFP8UnquantLoader
from text_generation_server.layers.marlin import GPTQMarlinWeightsLoader
from text_generation_server.layers.moe.gptq_marlin import (
    GPTQMarlinSparseMoELayer,
    can_use_marlin_moe_gemm,
)
from text_generation_server.layers.moe.unquantized import UnquantizedSparseMoELayer
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.log import log_once
from text_generation_server.utils.weights import (
    DefaultWeightsLoader,
    Weights,
    UnquantizedWeight,
)

if SYSTEM == "rocm":
    from .fused_moe_rocm import grouped_topk
    from vllm.model_executor.layers.fused_moe import fused_topk
elif SYSTEM != "ipex":
    from moe_kernels.fused_moe import fused_topk, grouped_topk


# NOTE: we are using a protocol here, because multiple inherance is not nice.
#       We need `Module`, and `Module` -> some abstract class -> some concrete
#       class inheritance is whacky.


@runtime_checkable
class MoELayer(Protocol):
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
        hidden_act: str = "silu",
    ): ...

    def forward(
        self, x: torch.Tensor, *, gating_output: torch.Tensor
    ) -> torch.Tensor: ...


class DenseMoELayer(nn.Module):
    """
    Layer for MoE that applies *all* experts to each tokens and then weights
    their outputs based on the calculated routing. This layer is much slower
    than `SparseMoELayer` and should only be used when no fused kernels are
    available (e.g. for unsupported quantizers).
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
        hidden_act: str = "silu",
    ):
        super().__init__()

        log_once(
            logger.info,
            "No fused layers are available for this model type, using (slower) dense MoE layer",
        )

        assert (n_expert_group is None) == (
            topk_group is None
        ), "n_expert_group and topk_group must both be None or have some value"

        self.n_expert_group = n_expert_group
        self.n_experts = n_experts
        self.renormalize = renormalize
        self.topk = topk
        self.topk_group = topk_group

        if "gelu" in hidden_act:
            self.act = lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh"
                    if hidden_act in ["gelu_fast", "gelu_pytorch_tanh"]
                    else "none"
                ),
            )
        elif "silu" in hidden_act:
            self.act = torch.nn.functional.silu
        else:
            self.act = ACT2FN[hidden_act]

        self.gate_proj = [
            TensorParallelColumnLinear.load(
                None,
                prefix=f"{prefix}.{i}.{gate_proj_name}",
                weights=weights,
                bias=False,
            )
            for i in range(self.n_experts)
        ]
        self.up_proj = [
            TensorParallelColumnLinear.load(
                None,
                prefix=f"{prefix}.{i}.{up_proj_name}",
                weights=weights,
                bias=False,
            )
            for i in range(self.n_experts)
        ]
        self.down_proj = [
            TensorParallelRowLinear.load(
                None,
                prefix=f"{prefix}.{i}.{down_proj_name}",
                weights=weights,
                bias=False,
            )
            for i in range(self.n_experts)
        ]

        self.process_group = weights.process_group

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gating_output: (sequence_length, n_experts)
        """
        # optional reshape
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        if self.n_expert_group is not None and self.topk_group is not None:
            topk_weights, topk_ids = grouped_topk(
                x,
                gating_output,
                self.topk,
                renormalize=self.renormalize,
                num_expert_group=self.n_expert_group,
                topk_group=self.topk_group,
            )
        else:
            topk_weights, topk_ids = fused_topk(
                x, gating_output, self.topk, self.renormalize
            )
            topk_weights = topk_weights.to(x.dtype)

        weights = torch.zeros(
            topk_ids.shape[0], self.n_experts, dtype=x.dtype, device=x.device
        )

        weights.scatter_(1, topk_ids.long(), topk_weights.to(weights.dtype))

        out = torch.zeros_like(x)
        for i in range(self.n_experts):
            h = self.act(self.gate_proj[i](x)) * self.up_proj[i](x)
            h = self.down_proj[i](h, reduce=False)
            out += h * weights[:, i].view(-1, 1)

        return out


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
        elif isinstance(
            weights.loader, GPTQMarlinWeightsLoader
        ) and can_use_marlin_moe_gemm(
            quant_method=weights.loader.quant_method,
            quantize=weights.loader.quantize,
            sym=weights.loader.sym,
        ):
            cls = GPTQMarlinSparseMoELayer
        else:
            raise ValueError(
                f"Unsupported weights loader: {type(weights.loader)}, sparse MoE is only supported for unquantized, AWQ, and GPTQ weights"
            )

        log_once(
            logger.info,
            "Using MoE layer wih fused gemm",
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
            or (
                isinstance(weights.loader, GPTQMarlinWeightsLoader)
                and can_use_marlin_moe_gemm(
                    quant_method=weights.loader.quant_method,
                    quantize=weights.loader.quantize,
                    sym=weights.loader.sym,
                )
            )
        )
