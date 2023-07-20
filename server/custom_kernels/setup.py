from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="custom_kernels",
    ext_modules=[
        CUDAExtension(
            name="custom_kernels.fused_bloom_attention_cuda",
            sources=["custom_kernels/fused_bloom_attention_cuda.cu"],
            extra_compile_args=["-arch=compute_80", "-std=c++17"],
        ),
        CUDAExtension(
            name="custom_kernels.fused_attention_cuda",
            sources=["custom_kernels/fused_attention_cuda.cu"],
            extra_compile_args=["-arch=compute_80", "-std=c++17"],
        ),
        CUDAExtension(
            name="custom_kernels.exllama",
            sources=[
                "custom_kernels/exllama/exllama_ext.cpp",
                "custom_kernels/exllama/cuda_buffers.cu",
                "custom_kernels/exllama/cuda_func/column_remap.cu",
                "custom_kernels/exllama/cuda_func/q4_matmul.cu",
                "custom_kernels/exllama/cuda_func/q4_matrix.cu"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
