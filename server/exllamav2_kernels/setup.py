from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

extra_cuda_cflags = ["-lineinfo", "-O3"]
extra_cflags = []
if torch.version.hip:
    extra_cflags = ["-DLEGACY_HIPBLAS_DIRECT=ON"]
    extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF", "-DLEGACY_HIPBLAS_DIRECT=ON"]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

setup(
    name="exllamav2_kernels",
    ext_modules=[
        CUDAExtension(
            name="exllamav2_kernels",
            sources=[
                "exllamav2_kernels/ext.cpp",
                "exllamav2_kernels/cuda/q_matrix.cu",
                "exllamav2_kernels/cuda/q_gemm.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
