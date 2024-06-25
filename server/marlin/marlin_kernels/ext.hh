#pragma once

#include <torch/library.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// No support for async
#else

torch::Tensor gptq_marlin_gemm(torch::Tensor &a, torch::Tensor &b_q_weight,
                               torch::Tensor &b_scales, torch::Tensor &g_idx,
                               torch::Tensor &perm, torch::Tensor &workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full);

torch::Tensor gptq_marlin_24_gemm(torch::Tensor &a, torch::Tensor &b_q_weight,
                                  torch::Tensor &b_meta,
                                  torch::Tensor &b_scales,
                                  torch::Tensor &workspace, int64_t num_bits,
                                  int64_t size_m, int64_t size_n,
                                  int64_t size_k);

torch::Tensor gptq_marlin_repack(torch::Tensor &b_q_weight, torch::Tensor &perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

torch::Tensor marlin_gemm(torch::Tensor &a, torch::Tensor &b_q_weight,
                          torch::Tensor &b_scales, torch::Tensor &workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k);

#endif
