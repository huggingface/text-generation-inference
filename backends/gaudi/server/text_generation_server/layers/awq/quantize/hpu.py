from typing import Optional
import torch
import torch.nn as nn

try:
    import habana_frameworks.torch.hpu  # noqa: F401

    convert_from_uint4 = torch.ops.hpu.convert_from_uint4
except Exception as e:
    hpu_import_exception = e

    def error_raiser_hpu(*args, **kwargs):
        raise ValueError(
            f"Trying to use HPU, but could not import the HPU framework with the following error: {hpu_import_exception}"
        )

    convert_from_uint4 = error_raiser_hpu

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(
            qzeros[:, :, None], shifts[None, None, :]
        ).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def unpack_weight_and_zeros(qweight, qzeros, bits):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    return iweight, izeros


def pack_tensor(input, bits=4):
    normal = input.to(torch.int32)
    q = torch.zeros(
        (normal.shape[0], normal.shape[1] // 32 * bits),
        dtype=torch.int32,
        device=input.device,
    )
    i = 0
    col = 0
    while col < q.shape[1]:
        for j in range(i, i + (32 // bits)):
            q[:, col] |= normal[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1
    q = q.to(torch.int32)
    return q


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
        self._preprocessing()

    def _preprocessing(self):
        device = self.qweight.device
        weight, zeros = unpack_weight_and_zeros(
            self.qweight.cpu(), self.qzeros.cpu(), self.w_bit
        )
        self.qweight = pack_tensor(weight).to(device)
        self.qzeros = pack_tensor(zeros).to(device)

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        weights = convert_from_uint4(self.qweight, self.scales, self.qzeros, x.dtype)
        outputs = torch.matmul(x, weights)

        outputs = outputs + self.bias if self.bias is not None else outputs
        outputs = outputs.reshape(out_shape)
        return outputs
