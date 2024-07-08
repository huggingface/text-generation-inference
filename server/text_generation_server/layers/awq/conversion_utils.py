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
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=imatrix.device)

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

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


def fast_awq_to_gptq(qweight, qzeros):
    # awq uses column packing for both weights and zeros
    izeros = unpack(qzeros, direction="column")
    iweights = unpack(qweight, direction="column")

    # Reverse the order of the iweight and izeros tensors
    izeros = apply_order(izeros, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    iweights = apply_order(iweights, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    # Subtract 1 from the izeros tensor (gptq adds 1 to the zeros)
    izeros = izeros - 1
    # exllama uses row packing for weights and column packing for zeros
    qzeros = pack(izeros, direction="column")
    qweight = pack(iweights, direction="row")

    return qweight, qzeros
