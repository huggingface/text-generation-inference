//
// Created by Morgan Funtowicz on 6/30/24.
//

#ifndef TGI_TRTLLM_BACKEND_H
#define TGI_TRTLLM_BACKEND_H

#include <filesystem>

//#include <tensorrt_llm/runtime/common.h>
//#include <tensorrt_llm/executor/executor.h>
//
//namespace tle = tensorrt_llm::executor;

namespace huggingface::tgi::backends {
    class TensorRtLlmBackend {
    private:
//        tle::Executor executor;

    public:
        TensorRtLlmBackend(const std::filesystem::path &engineFolder);
    };
}

#endif //TGI_TRTLLM_BACKEND_H
