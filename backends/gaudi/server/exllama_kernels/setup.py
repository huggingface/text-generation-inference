from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

extra_cuda_cflags = []
extra_cflags = []
if torch.version.hip:
    extra_cflags = ["-DLEGACY_HIPBLAS_DIRECT=ON"]
    extra_cuda_cflags = ["-DLEGACY_HIPBLAS_DIRECT=ON"]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

setup(
    name="exllama_kernels",
    ext_modules=[
        CUDAExtension(
            name="exllama_kernels",
            sources=[
                "exllama_kernels/exllama_ext.cpp",
                "exllama_kernels/cuda_buffers.cu",
                "exllama_kernels/cuda_func/column_remap.cu",
                "exllama_kernels/cuda_func/q4_matmul.cu",
                "exllama_kernels/cuda_func/q4_matrix.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
