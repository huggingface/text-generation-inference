#include <torch/extension.h>

#include "ext.hh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("awq_marlin_repack", &awq_marlin_repack,
        "Repack AWQ parameters for Marlin");
  m.def("gptq_marlin_gemm", &gptq_marlin_gemm,
        "Marlin gemm with GPTQ compatibility");
  m.def("gptq_marlin_24_gemm", &gptq_marlin_24_gemm, "Marlin sparse 2:4 gemm");
  m.def("gptq_marlin_repack", &gptq_marlin_repack,
        "Repack GPTQ parameters for Marlin");
  m.def("marlin_gemm", &marlin_gemm, "Marlin gemm");
  // fp8_marlin Optimized Quantized GEMM for FP8 weight-only.
  m.def("fp8_marlin_gemm", &fp8_marlin_gemm);
}
