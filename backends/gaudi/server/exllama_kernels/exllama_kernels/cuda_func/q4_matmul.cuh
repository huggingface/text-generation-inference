// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _q4_matmul_cuh
#define _q4_matmul_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q4_matrix.cuh"
#include "../tuning.h"

void q4_matmul_cuda
(
    ExLlamaTuning* tuningParams,
    const half* x,
    const int x_height,
    const Q4Matrix* w,
    half* out,
    bool no_zero,
    cudaStream_t alt_stream
);

void q4_matmul_recons_cuda
(
    ExLlamaTuning* tuningParams,
    const half* x,
    const int x_height,
    Q4Matrix* w,
    half* out,
    bool no_zero,
    const cublasHandle_t handle
);

#endif
