import math
import numpy as np
import torch
import torch.nn as nn

import intel_extension_for_pytorch as ipex


class QuantLinear(nn.Module):
    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("qzeros", qzeros)
        self.register_buffer("scales", scales)
        self.register_buffer("g_idx", g_idx)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize

        self.outfeatures = qweight.shape[1]
        self.infeatures = qweight.shape[0] * 32 // bits
        self.woq_linear = (
            ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(
                self.qweight,
                self.scales,
                self.qzeros,
                self.infeatures,
                self.outfeatures,
                bias=self.bias,
                group_size=self.groupsize,
                g_idx=g_idx,
                quant_method=ipex.llm.quantization.QuantMethod.GPTQ_GEMM,
                dtype=ipex.llm.quantization.QuantDtype.INT4,
            )
        )

    @classmethod
    def new(cls, bits, groupsize, infeatures, outfeatures, bias):
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")

        qweight = torch.zeros((infeatures // 32 * bits, outfeatures), dtype=torch.int32)
        qzeros = torch.zeros(
            (math.ceil(infeatures / groupsize), outfeatures // 32 * bits),
            dtype=torch.int32,
        )
        scales = torch.zeros(
            (math.ceil(infeatures / groupsize), outfeatures), dtype=torch.float16
        )
        g_idx = torch.tensor(
            [i // groupsize for i in range(infeatures)], dtype=torch.int32
        )
        if bias:
            bias = torch.zeros((outfeatures), dtype=torch.float16)
        else:
            bias = None
        return cls(qweight, qzeros, scales, g_idx, bias, bits, groupsize)

    def pack(self, linear, scales, zeros, g_idx=None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]])
                    / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [4]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 4 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32
        )
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [4]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 4 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        out = self.woq_linear(x.reshape(-1, x.shape[-1]))
        return out.reshape(out_shape)
