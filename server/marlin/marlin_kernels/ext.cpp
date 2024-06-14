#include <torch/extension.h>

#include "ext.hh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gptq_marlin_gemm", &gptq_marlin_gemm,
        "Marlin gemm with GPTQ compatibility");
  m.def("gptq_marlin_repack", &gptq_marlin_repack,
        "Repack GPTQ parameters for Marlin");
  m.def("marlin_gemm", &marlin_gemm, "Marlin gemm");
}
