//
// Created by mfuntowicz on 6/30/24.
//
#include <filesystem>
#include "rust/cxx.h"

#include "backends/trtllm/include/backend.h"

namespace huggingface::tgi::backends {
    /***
    *
    * @param engineFolder
    * @return
    */
    std::unique_ptr<TensorRtLlmBackend> create_trtllm_backend(rust::Str engineFolder) {
        const auto enginePath = std::string_view(engineFolder.begin(), engineFolder.end());
        return std::make_unique<TensorRtLlmBackend>(enginePath);
    }

}