#include <spdlog/spdlog.h>
#include <fmt/std.h>

#include "backend.h"

huggingface::tgi::backends::TensorRtLlmBackendImpl::TensorRtLlmBackendImpl(std::filesystem::path &engineFolder) {
    SPDLOG_INFO(FMT_STRING("Loading engines from {}"), engineFolder);
}

std::unique_ptr<huggingface::tgi::backends::TensorRtLlmBackendImpl>
huggingface::tgi::backends::create_trtllm_backend(std::filesystem::path &engineFolder) {
    return std::make_unique<huggingface::tgi::backends::TensorRtLlmBackendImpl>(engineFolder);
}
