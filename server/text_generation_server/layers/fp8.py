import torch
from .marlin import _check_marlin_kernels, marlin_kernels, permute_scales

def fp8_quantize(weight, qdtype=torch.float8_e4m3fn):
    device = weight.device
    # weight, scale = quant_weights(weight, torch.int8, False)
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale


def pack_fp8_to_int32(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.shape[0] % 4 == 0

    # Reshape to prepare for packing
    reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])

    # Convert fp8 to uint8 (byte) representation
    byte_tensor = reshaped.view(torch.uint8)

    # Pack 4 uint8 values into one int32
    packed = (byte_tensor[:, 0].to(torch.int32) |
              (byte_tensor[:, 1].to(torch.int32) << 8) |
              (byte_tensor[:, 2].to(torch.int32) << 16) |
              (byte_tensor[:, 3].to(torch.int32) << 24))

    return packed.view(fp8_tensor.shape[0] // 4,
                       *fp8_tensor.shape[1:]).contiguous()


def _repack(weight: torch.Tensor, scales):
    qweight = pack_fp8_to_int32(weight)
    perm = torch.empty(0, dtype=torch.int, device=qweight.device)
    marlin_kernels.gptq_marlin_repack(
        qweight, perm, in_features, out_features, 8
    )

    finfo = torch.finfo(qweight.dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / qweight.abs().max().clamp(min=1e-12)
    scale = scale.float().reciprocal()

    scale = permute_scales(scale)


class Fp8Linear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.dtype = weight.dtype
        self.qweight, self.scale = fp8_quantize(weight)

        self.bias = bias if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qinput, scale = fp8_quantize(input)
        output, _ = torch._scaled_mm(
            qinput,
            self.qweight.t(),
            out_dtype=self.dtype,
            scale_a=scale,
            scale_b=self.scale,
            bias=self.bias,
        )
        return output
