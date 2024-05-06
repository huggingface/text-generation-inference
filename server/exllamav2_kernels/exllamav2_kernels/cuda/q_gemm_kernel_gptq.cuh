#include "compat.cuh"

__forceinline__ __device__ half2 dot22_8(half2(&dq)[4], const half* a_ptr, const half2 g_result)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hadd2(result, g_result);
}

__forceinline__ __device__ float dot22_8_f(half2(&dq)[4], const half* a_ptr)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __half2float(__low2half(result)) + __half2float(__high2half(result));
}

__forceinline__ __device__ half2 dot22_8_h2(half2(&dq)[4], const half* a_ptr)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return result;
}

typedef void (*fp_gemm_half_q_half_gptq_kernel)
(
    const half*,
    const uint32_t*,
    const uint32_t*,
    const half*,
    half*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const uint16_t*,
    const int,
    const bool,
    const half*,
    const int
);

template <int m_count, bool use_r_weights, bool mul_r_weights>
__global__ void gemm_half_q_half_gptq_kernel
(
    const half* __restrict__ a,
    const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    half* __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const int groupsize,
    const uint16_t* __restrict__ b_q_perm,
    const int rows_4,
    const bool clear,
    const half* r_weights,
    const int r_weights_stride
)
{
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int t = threadIdx.x;

    // Block

    int offset_n = blockIdx.x * GPTQ_BLOCK_KN_SIZE * 4;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * GPTQ_BLOCK_KN_SIZE;

    int end_n = min(offset_n + GPTQ_BLOCK_KN_SIZE * 4, size_n);
    int end_m = min(offset_m + m_count, size_m);
    int end_k = min(offset_k + GPTQ_BLOCK_KN_SIZE, size_k);

    int n = offset_n + t * 4;

    // Read weights

    half_uint16 weights[MAX_Q_GEMM_WEIGHTS];
    if constexpr (use_r_weights)
    {
        uint16_t any_w = 0;
        const half* w_ptr = r_weights;
        for (int m = 0; m < m_count; ++m)
        {
            weights[m].as_half = *w_ptr;
            w_ptr += r_weights_stride;
            any_w |= weights[m].as_uint16;
        }
        if (!any_w) return;  // Early exit if all weights are zero -- does not zero output (!!!)
    }

    // Preload block_a

    __shared__ half block_a[m_count][GPTQ_BLOCK_KN_SIZE];

    if (offset_k + t < end_k)
    {
        for (int m = 0; m < m_count; ++m)
        {
            const half* a_ptr = a_.item_ptr(offset_m + m, 0);
            half* block_a_ptr = block_a[m];

            half a0;
            if (b_q_perm) a0 = a_ptr[b_q_perm[offset_k + t]];
            else a0 = a_ptr[offset_k + t];
            block_a_ptr[t] = a0;
        }
    }

    // Zero output

    if (n >= size_n) return;

    if (clear && blockIdx.z == 0) // && (threadIdx.x & 1) == 0)
    {
        for (int m = 0; m < m_count; m++)
            *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
    }

    __syncthreads();

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // a, b offset

    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
    const half* a_ptr = &block_a[0][0];
    int a_stride = GPTQ_BLOCK_KN_SIZE;

    // Initial group

    int zeros[4];
    half2 scales[4];
    half2 z1z16[4][2];
    half2 y1y16[4][2];
    b_gptq_qzeros_.item4(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);
    dequant_4bit_8_prep_zero((zeros[0] + 1) & 0x0F, z1z16[0], y1y16[0]);
    dequant_4bit_8_prep_zero((zeros[1] + 1) & 0x0F, z1z16[1], y1y16[1]);
    dequant_4bit_8_prep_zero((zeros[2] + 1) & 0x0F, z1z16[2], y1y16[2]);
    dequant_4bit_8_prep_zero((zeros[3] + 1) & 0x0F, z1z16[3], y1y16[3]);

//    __syncthreads();

    // Column result

    half2 block_c[m_count][4] = {};

    // Dequantize and multiply

    int k = offset_k;
    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_qzeros_.item4(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
            dequant_4bit_8_prep_zero((zeros[0] + 1) & 0x0F, z1z16[0], y1y16[0]);
            dequant_4bit_8_prep_zero((zeros[1] + 1) & 0x0F, z1z16[1], y1y16[1]);
            dequant_4bit_8_prep_zero((zeros[2] + 1) & 0x0F, z1z16[2], y1y16[2]);
            dequant_4bit_8_prep_zero((zeros[3] + 1) & 0x0F, z1z16[3], y1y16[3]);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const int4* b_ptr4 = (int4*) b_ptr;
            int4 load_int4 = *b_ptr4;

            half2 dq[4][4];
            dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n, false);
            dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n, false);
            dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n, false);
            dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n, false);

            #pragma unroll
            for (int m = 0; m < m_count; m++)
            {
                if constexpr (use_r_weights) { if (!weights[m].as_uint16) continue; }
                block_c[m][0] = __hfma2(dot22_8_h2(dq[0], a_ptr + m * a_stride), scales[0], block_c[m][0]);
                block_c[m][1] = __hfma2(dot22_8_h2(dq[1], a_ptr + m * a_stride), scales[1], block_c[m][1]);
                block_c[m][2] = __hfma2(dot22_8_h2(dq[2], a_ptr + m * a_stride), scales[2], block_c[m][2]);
                block_c[m][3] = __hfma2(dot22_8_h2(dq[3], a_ptr + m * a_stride), scales[3], block_c[m][3]);
            }

            b_ptr += size_n;
            a_ptr += 8;
        }

        k += 32;
    }

    for (int m = 0; m < m_count; m++)
    {
        half2 *out = (half2*) c_.item_ptr(offset_m + m, n);
        half result0 = __hadd(__low2half(block_c[m][0]), __high2half(block_c[m][0]));
        half result1 = __hadd(__low2half(block_c[m][1]), __high2half(block_c[m][1]));
        half result2 = __hadd(__low2half(block_c[m][2]), __high2half(block_c[m][2]));
        half result3 = __hadd(__low2half(block_c[m][3]), __high2half(block_c[m][3]));
        half2 result01 = __halves2half2(result0, result1);
        half2 result23 = __halves2half2(result2, result3);

        if constexpr (mul_r_weights)
        {
            half2 w_mul2 = __half2half2(weights[m].as_half);
            result01 = __hmul2(result01, w_mul2);
            result23 = __hmul2(result23, w_mul2);
        }

        atomicAdd(out    , result01);
        atomicAdd(out + 1, result23);
    }
}

template <bool use_r_weights, bool mul_r_weights>
struct map_m_count_gptq {
    static constexpr fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(int m_count)
    {
        #if GPTQ_BLOCK_M_SIZE_MAX >= 1
        if (m_count == 1) return gemm_half_q_half_gptq_kernel<1, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 2
        if (m_count == 2) return gemm_half_q_half_gptq_kernel<2, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 3
        if (m_count == 3) return gemm_half_q_half_gptq_kernel<3, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 4
        if (m_count == 4) return gemm_half_q_half_gptq_kernel<4, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 5
        if (m_count == 5) return gemm_half_q_half_gptq_kernel<5, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 6
        if (m_count == 6) return gemm_half_q_half_gptq_kernel<6, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 7
        if (m_count == 7) return gemm_half_q_half_gptq_kernel<7, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 8
        if (m_count == 8) return gemm_half_q_half_gptq_kernel<8, use_r_weights, mul_r_weights>;
        #endif
        return NULL;
    }
};

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(const int m_count, bool r_weights, bool mul_r_weights)
{
    if (!r_weights && !mul_r_weights) return map_m_count_gptq<false, false>::pick_gemm_half_q_half_gptq_kernel(m_count);
    if (!r_weights &&  mul_r_weights) return map_m_count_gptq<false,  true>::pick_gemm_half_q_half_gptq_kernel(m_count);
    if ( r_weights && !mul_r_weights) return map_m_count_gptq< true, false>::pick_gemm_half_q_half_gptq_kernel(m_count);
    if ( r_weights &&  mul_r_weights) return map_m_count_gptq< true,  true>::pick_gemm_half_q_half_gptq_kernel(m_count);
    return NULL;
}
