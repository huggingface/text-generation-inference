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
    std::unique_ptr<TensorRtLlmBackend> create_trtllm_backend(rust::Str engineFolder, rust::Str executorWorker) {
        // Unconditionally call this to initialize and discover TRTLLM plugins
        InitializeBackend();

        const auto enginePath = std::string_view(engineFolder.begin(), engineFolder.end());
        const auto executorPath = std::string_view(executorWorker.begin(), executorWorker.end());
        return std::make_unique<TensorRtLlmBackend>(std::move(enginePath), std::move(executorPath));
    }
}