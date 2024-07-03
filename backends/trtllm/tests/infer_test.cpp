//
// Created by mfuntowicz on 7/2/24.
//
#include <catch2/catch_all.hpp>
#include "../include/backend.h"

TEST_CASE("Load TRTLLM Engine on the TGI Backend", "[trtllm][engine][load]") {
    huggingface::tgi::backends::TensorRtLlmBackend backend("fixtures/engines/llama3-8b-instruct.engine");
}