from dataclasses import dataclass

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Int8Params, Params4bit
from text_generation_server.utils.weights import UnquantizedWeight


@dataclass
class BNBWeight(UnquantizedWeight):
    weight: torch.Tensor

    def get_linear(self, bias: torch.Tensor):
        return Linear8bitLt(self.weight, bias, has_fp16_weights=False, threshold=6.0)


class Linear8bitLt(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        has_fp16_weights=True,
        threshold=0.0,
        index=None,
    ):
        super().__init__()
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights

        self.weight = Int8Params(
            weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )
        self.weight.cuda(weight.device)
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights and self.state.CB is not None:
            self.weight.data = self.state.CB

        return out


@dataclass
class BNBFP4Weight(UnquantizedWeight):
    weight: torch.Tensor

    def get_linear(self, bias: torch.Tensor):
        return Linear4bit(self.weight, bias, quant_type="fp4")


@dataclass
class BNBNF4Weight(UnquantizedWeight):
    weight: torch.Tensor

    def get_linear(self, bias: torch.Tensor):
        return Linear4bit(self.weight, bias, quant_type="nf4")


class Linear4bit(torch.nn.Module):
    def __init__(self, weight, bias, quant_type):
        super().__init__()
        self.weight = Params4bit(
            weight.data,
            requires_grad=False,
            compress_statistics=True,
            quant_type=quant_type,
        )
        self.compute_dtype = None
        self.weight.cuda(weight.device)
        self.bias = bias

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)

        return bnb.matmul_4bit(
            x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state
        ).to(inp_dtype)
