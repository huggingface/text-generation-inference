from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='my_custom_comm',
      include_dirs=["/usr/local/cuda/targets/x86_64-linux/include/",
                    ],
      ext_modules=[CUDAExtension('my_custom_comm',
                                 ['my_custom_comm.cc'],
                                 libraries=["mpi"]
                                 ),],
      cmdclass={'build_ext': BuildExtension},
      )
