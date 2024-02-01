import torch
from typing import List


AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def pack(imatrix: torch.Tensor, direction: str = "column"):
    """
    Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of packing, either "column" or "row"
    Returns:
        qmatrix (torch.Tensor): packed matrix of integers
    """
    shifts = torch.arange(0, 32, 4, device=imatrix.device)

    imatrix = imatrix.to(torch.int8)
    imatrix = torch.bitwise_and(imatrix, 0x0F)  # eventually correct overflow

    if direction == "column":
        imatrix = imatrix.view(-1, imatrix.shape[1] // (32 // 4), (32 // 4))
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        imatrix = imatrix.view(imatrix.shape[0] // (32 // 4), (32 // 4), -1)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)

    return qmatrix


def unpack(qmatrix: torch.Tensor, direction: str = "column"):
    """
    Unpacks a 32-bit packed integer matrix into a 4-bit integer matrix.
    Args:
        qmatrix (torch.Tensor): matrix of packed integers
        direction (str): direction of unpacking, either "column" or "row"
    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    shifts = torch.arange(0, 32, 4, device=qmatrix.device)

    if direction == "column":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, :, None], shifts[None, None, :]
        ).view(qmatrix.shape[0], -1)

    elif direction == "row":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, None, :], shifts[None, :, None]
        ).view(-1, qmatrix.shape[-1])

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    return imatrix


def quantize(fmatrix, scales, zeros, group_size):
    """
    Quantizes a matrix of 16-bit floats into a matrix of 4-bit integers.
    Args:
        fmatrix (torch.Tensor): matrix of 16-bit floats
        scales (torch.Tensor): matrix of 16-bit floats
        zeros (torch.Tensor): matrix of 4-bit integers
        group_size (int): group size
    Returns:
        imatrix (torch.Tensor): matrix of 4-bit integers
    """
    zeros = zeros.to(torch.int8) & 0x0F

    imatrix = torch.round(
        (
            fmatrix / scales.repeat_interleave(group_size, dim=0)
            + zeros.repeat_interleave(group_size, dim=0)
        )
    )

    imatrix = imatrix.to(torch.int8) & 0x0F

    return imatrix


def dequantize(imatrix, scales, zeros, group_size):
    """
    Dequantizes a 4-bit integer matrix into a float matrix.
    Args:
        imatrix (torch.Tensor): matrix of 4-bit integers
        scales (torch.Tensor): matrix of 16-bit floats
        zeros (torch.Tensor): matrix of 4-bit integers
        group_size (int): group size
    Returns:
        fmatrix (torch.Tensor): matrix of 16-bit floats
    """
    zeros = zeros.to(torch.int8) & 0x0F
    imatrix = imatrix.to(torch.int8) & 0x0F

    fmatrix = (
        imatrix - zeros.repeat_interleave(group_size, dim=0)
    ) * scales.repeat_interleave(group_size, dim=0)

    fmatrix = fmatrix.to(torch.float16)

    return fmatrix


def apply_order(
    imatrix: torch.Tensor,
    direction: str = "column",
    order: List[int] = AWQ_PACK_ORDER,
):
    """
    Applies the order to a 4-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of applying order, either "column" or "row"
        order (List[int]): order to apply, default is AWQ_PACK_ORDER
    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    if direction == "column":
        imatrix = imatrix.view(-1, (32 // 4))[:, order].view(imatrix.shape)
    elif direction == "row":
        imatrix = imatrix.view((32 // 4), -1)[order, :].view(imatrix.shape)

    return imatrix


def fast_awq_to_exllama(qweight, qzeros):
    # awq uses column packing for both weights and zeros
    izeros = unpack(qzeros, direction="column")
    iweights = unpack(qweight, direction="column")

    # Reverse the order of the iweight and izeros tensors
    izeros = apply_order(izeros, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    iweights = apply_order(iweights, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    # Subtract 1 from the izeros tensor (exllama adds 1 during inference)
    izeros = izeros - 1
    # exllama uses row packing for weights and column packing for zeros
    qzeros = pack(izeros, direction="column")
    qweight = pack(iweights, direction="row")

    return qweight, qzeros
