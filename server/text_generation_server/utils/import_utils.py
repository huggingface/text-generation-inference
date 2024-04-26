import torch


def is_xpu_available():
    try:
        import intel_extension_for_pytorch
    except ImportError:
        return False

    return hasattr(torch, "xpu") and torch.xpu.is_available()


IS_ROCM_SYSTEM = torch.version.hip is not None
IS_CUDA_SYSTEM = torch.version.cuda is not None
IS_XPU_SYSTEM = is_xpu_available()
