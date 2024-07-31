//
// Created by mfuntowicz on 7/2/24.
//
#include <catch2/catch_all.hpp>
#include <spdlog/spdlog.h>
#include "../include/backend.h"

TEST_CASE("Load TRTLLM Engine on the TGI Backend", "[trtllm][engine][load]") {
    const auto engines = std::filesystem::path("/home/mfuntowicz/.cache/huggingface/assets/trtllm/0.11.0.dev2024062500/meta-llama--Meta-Llama-3-8B-Instruct/4090/engines/");
    const auto executor = std::filesystem::path("/home/mfuntowicz/Workspace/text-generation-inference/backends/trtllm/cmake-build-debug/cmake-build-debug/_deps/trtllm-src/cpp/tensorrt_llm/executor_worker/executorWorker");

    spdlog::info("Loading config from: {}", absolute(engines).string());
    huggingface::tgi::backends::TensorRtLlmBackend backend(engines, executor);
}
