import torch

def gptq_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    g_idx: torch.Tensor,
    perm: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
) -> torch.Tensor:
    """
    Matrix multiplication using Marlin kernels. This is an extension of
    `marlin_gemm` that supports converted GPTQ kernels.
    """
    ...

def gptq_marlin_24_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_meta: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    """
    Matrix multiplication using Marlin kernels. This is an extension of
    `marlin_gemm` that supports 2:4 sparsity.
    """
    ...

def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack GPTQ parameters for Marlin kernels."""
    ...

def marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    """
    Matrix multiplication using Marlin kernels.
    """
    ...

# fp8 marlin
def fp8_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return torch.ops._C.fp8_marlin_gemm(
        a, b_q_weight, b_scales, workspace, num_bits, size_m, size_n, size_k
    )
