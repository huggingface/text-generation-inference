import torch


def is_npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False

    return hasattr(torch, "npu") and torch.npu.is_available()


IS_ROCM_SYSTEM = torch.version.hip is not None
IS_CUDA_SYSTEM = torch.version.cuda is not None
IS_NPU_SYSTEM = is_npu_available()
