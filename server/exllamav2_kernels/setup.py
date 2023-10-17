from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="exllamav2_kernels",
    ext_modules=[
        CUDAExtension(
            name="exllamav2_kernels",
            sources=[
                "autogptq_extension/exllamav2/ext.cpp",
                "autogptq_extension/exllamav2/cuda/q_matrix.cu",
                "autogptq_extension/exllamav2/cuda/q_gemm.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
