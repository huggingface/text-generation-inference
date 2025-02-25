#include "q_gemm.cuh"
#include "util.cuh"
#include "matrix_view.cuh"
#include "../config.h"

#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"

#define GPTQ_BLOCK_KN_SIZE 128
#define GPTQ_BLOCK_M_SIZE_MAX 8
#define GPTQ_MAX_GROUPS_IN_BLOCK (GPTQ_BLOCK_KN_SIZE / 32)

#define EXL2_BLOCK_KN_SIZE 64
#define EXL2_BLOCK_M_SIZE_MAX 8
#define EXL2_MAX_GROUPS_IN_BLOCK (EXL2_BLOCK_KN_SIZE / 32)

#define CLEAR_N_SIZE 256

#include "q_gemm_kernel.cuh"
#include "q_gemm_kernel_gptq.cuh"

void gemm_half_q_half_cuda_part
(
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    int m_count,
    bool clear,
    const half* r_weights,
    int r_weights_stride,
    bool mul_r_weights
)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!b->is_gptq)
    {
        dim3 blockDim, gridDim;
        blockDim.x = EXL2_BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(size_n, EXL2_BLOCK_KN_SIZE * 4);
        gridDim.y = DIVIDE(size_m, m_count);
        gridDim.z = DIVIDE(size_k, EXL2_BLOCK_KN_SIZE);

        fp_gemm_half_q_half_kernel kernel = pick_gemm_half_q_half_kernel(m_count, r_weights != NULL, mul_r_weights);

        kernel<<<gridDim, blockDim, 0, stream>>>
        (
            a,
            b->cuda_q_weight,
            b->cuda_q_scale,
            b->cuda_q_scale_max,
            c,
            size_m,
            size_n,
            size_k,
            b->groups,
            b->cuda_q_group_map,
            b->cuda_q_perm,
            b->rows_8,
            b->rows_6,
            b->rows_5,
            b->rows_4,
            b->rows_3,
            b->rows_2,
            clear,
            r_weights,
            r_weights_stride
        );
    }
    else
    {
        dim3 blockDim, gridDim;
        blockDim.x = GPTQ_BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(size_n, GPTQ_BLOCK_KN_SIZE * 4);
        gridDim.y = DIVIDE(size_m, m_count);
        gridDim.z = DIVIDE(size_k, GPTQ_BLOCK_KN_SIZE);

        fp_gemm_half_q_half_gptq_kernel kernel = pick_gemm_half_q_half_gptq_kernel(m_count, r_weights != NULL, mul_r_weights);

//         DBGX((uint64_t) r_weights);
//         if (r_weights)
//             print_global_mem(r_weights, 1, 1, 1);
//         DBGI(r_weights_stride);

        kernel<<<gridDim, blockDim, 0, stream>>>
        (
            a,
            b->cuda_q_weight,
            b->cuda_gptq_qzeros,
            b->cuda_gptq_scales,
            c,
            size_m,
            size_n,
            size_k,
            b->groups,
            b->gptq_groupsize,
            b->cuda_q_perm,
            b->rows_4,
            clear,
            r_weights,
            r_weights_stride
        );
    }
}

void gemm_half_q_half_cuda
(
    cublasHandle_t cublas_handle,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    bool clear,
    half* temp_dq,
    bool force_cuda,
    const half* r_weights,
    const int r_weights_stride,
    bool mul_r_weights
)
{
    if (size_m > MAX_Q_GEMM_ROWS && !force_cuda)
    {
        // Reconstruct FP16 matrix, then cuBLAS

        if (!temp_dq) temp_dq = b->temp_dq;
        b->reconstruct(temp_dq);

        //cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

        const half alpha = __float2half(1.0f);
        const half beta = clear ? __float2half(0.0f) : __float2half(1.0f);
        cublasHgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    size_n, size_m, size_k,
                    &alpha, temp_dq, size_n,
                            a,       size_k,
                    &beta,  c,       size_n);

        //const float alpha = 1.0f;
        //const float beta = clear ? 0.0f : 1.0f;
        //cublasSgemmEx(cublas_handle,
        //             CUBLAS_OP_N,
        //             CUBLAS_OP_N,
        //             size_n, size_m, size_k,
        //             &alpha, temp_dq, CUDA_R_16F, size_n,
        //                     a,       CUDA_R_16F, size_k,
        //             &beta,  c,       CUDA_R_16F, size_n);

        //const float alpha = 1.0f;
        //const float beta = clear ? 0.0f : 1.0f;
        //cublasGemmEx(cublas_handle,
        //             CUBLAS_OP_N, CUBLAS_OP_N,
        //             size_n, size_m, size_k,
        //             &alpha, temp_dq, CUDA_R_16F, size_n,
        //                     a,       CUDA_R_16F, size_k,
        //             &beta,  c,       CUDA_R_16F, size_n,
        //             CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP);
    }
    else
    {
        // Quantized matmul

        int block_m_size_max = b->is_gptq ? GPTQ_BLOCK_M_SIZE_MAX : EXL2_BLOCK_M_SIZE_MAX;
        int max_chunks = size_m / block_m_size_max;
        int last_chunk = max_chunks * block_m_size_max;
        int last_chunk_size = size_m - last_chunk;

        if (max_chunks)
        {
            gemm_half_q_half_cuda_part(a, b, c, last_chunk, size_n, size_k, block_m_size_max, clear, r_weights, r_weights_stride, mul_r_weights);
        }

        if (last_chunk_size)
        {
            gemm_half_q_half_cuda_part(a + last_chunk * size_k, b, c + last_chunk * size_n, last_chunk_size, size_n, size_k, last_chunk_size, clear, r_weights, r_weights_stride, mul_r_weights);
        }
    }
}

__global__ void clear_kernel
(
    half* __restrict__ c,
    const int size_m,
    const int size_n
)
{
    int m = blockIdx.y;
    int n = (blockIdx.x * CLEAR_N_SIZE + threadIdx.x) * 8;
    if (n >= size_n) return;
    int4* c_ptr = (int4*)(c + m * size_n + n);
    *c_ptr = {};
}

void clear_tensor_cuda
(
    half* c,
    int size_m,
    int size_n
)
{
//     dim3 blockDim, gridDim;
//     blockDim.x = CLEAR_N_SIZE;
//     blockDim.y = 1;
//     gridDim.x = DIVIDE(size_n / 8, CLEAR_N_SIZE);
//     gridDim.y = size_m;
//     clear_kernel<<<gridDim, blockDim>>>(c, size_m, size_n);
}
