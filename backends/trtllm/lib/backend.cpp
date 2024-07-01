#include <spdlog/spdlog.h>
#include <fmt/std.h>

#include "backend.h"

huggingface::tgi::backends::TensorRtLlmBackend::TensorRtLlmBackend(const std::filesystem::path &engineFolder) {
    SPDLOG_INFO(FMT_STRING("Loading engines from {}"), engineFolder);
}
