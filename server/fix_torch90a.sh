#!/bin/bash

# This script is required to patch torch < 2.4
# It adds the 90a cuda target (H100)
# This target is required to build FBGEMM kernels

torch_cuda_arch=$(python -c "import torch; print(torch.__file__)" | sed 's/\/__init__.py//; s|$|/share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake|')

sed -i '189s/\[0-9]\\\\\.\[0-9](/[0-9]\\\\.[0-9]a?(/' $torch_cuda_arch
sed -i '245s/\[0-9()]+\+"/[0-9()]+a?"/' $torch_cuda_arch
sed -i '246s/\[0-9]+\+"/[0-9]+a?"/' $torch_cuda_arch
