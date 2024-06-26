from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = []

setup(
    name="marlin_kernels",
    ext_modules=[
        CUDAExtension(
            name="marlin_kernels",
            sources=[
                "marlin_kernels/gptq_marlin.cu",
                "marlin_kernels/gptq_marlin_repack.cu",
                "marlin_kernels/marlin_cuda_kernel.cu",
                "marlin_kernels/sparse/marlin_24_cuda_kernel.cu",
                "marlin_kernels/ext.cpp",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
