import math
import numpy as np
import torch
import torch.nn as nn

try:

    convert_from_uint4 = torch.ops.hpu.convert_from_uint4
except Exception as e:
    hpu_import_exception = e

    def error_raiser_hpu(*args, **kwargs):
        raise ValueError(
            f"Trying to use HPU, but could not import the HPU framework with the following error: {hpu_import_exception}"
        )

    convert_from_uint4 = error_raiser_hpu


def pack_tensor(input, bits=4):
    normal = input.to(torch.int32)
    q = torch.zeros((normal.shape[0], normal.shape[1] // 32 * bits), dtype=torch.int32)
    i = 0
    col = 0
    while col < q.shape[1]:
        for j in range(i, i + (32 // bits)):
            q[:, col] |= normal[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1
    q = q.to(torch.int32)
    return q


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
        self.wf = torch.tensor(
            list(range(0, 32, self.bits)), dtype=torch.int32
        ).unsqueeze(0)
        self._preprocessing()

    def unpack_zeros_from_cuda_old_format(self):
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1).to(
            self.scales.dtype
        )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.
        zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
        return zeros

    def unpack_weight_from_cuda_old_format(self):
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape((weight.shape[0] * weight.shape[1], weight.shape[2]))
        return weight

    def _preprocessing(self):
        orig_device = self.qweight.device
        self.qweight = self.qweight.cpu()
        weight = self.unpack_weight_from_cuda_old_format()
        new_qweight = pack_tensor(weight)
        self.qweight = new_qweight.to(orig_device)
        # TODO: Support group indexing and remove the check
        columns = self.qweight.shape[0]
        g_idx_trivial = [i // self.groupsize for i in range(columns)]
        g_idx_trivial = torch.tensor(
            g_idx_trivial, dtype=torch.int32, device=self.g_idx.device
        )
        assert torch.equal(
            self.g_idx, g_idx_trivial
        ), "Non-trivial tensor g_idx is not supported"
        self.qzeros = self.qzeros.cpu()
        zeros = self.unpack_zeros_from_cuda_old_format()
        new_qzeros = pack_tensor(zeros)
        self.qzeros = new_qzeros.to(orig_device)

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
        x = x.reshape(-1, x.shape[-1])
        weight = convert_from_uint4(self.qweight, self.scales, self.qzeros, x.dtype)
        out = torch.matmul(x, weight)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out
