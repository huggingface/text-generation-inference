from typing import Optional
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex


class WQLinear(nn.Module):
    def __init__(
        self, w_bit, group_size, qweight, qzeros, scales, bias: Optional[torch.Tensor]
    ):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 32 // w_bit

        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else self.in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert self.out_features % (32 // self.w_bit) == 0

        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bias = bias
        self.woq_linear = (
            ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(
                self.qweight,
                self.scales,
                self.qzeros,
                self.in_features,
                self.out_features,
                bias=self.bias,
                group_size=self.group_size,
                quant_method=ipex.llm.quantization.QuantMethod.AWQ_GEMM,
                dtype=ipex.llm.quantization.QuantDtype.INT4,
            )
        )

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        out = self.woq_linear(x.reshape(-1, x.shape[-1]))
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
