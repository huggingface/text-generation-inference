//
// Created by mfuntowicz on 7/11/24.
//

#ifndef TGI_TRTLLM_BACKEND_FFI_H
#define TGI_TRTLLM_BACKEND_FFI_H

#include <cmath>
#include <cstddef>
#include <memory>
#include "backend.h"

namespace huggingface::tgi::backends {
    class TensorRtLlmBackendImpl;
}

// Template to support returning error from TllmException back to Rust in a Result<>
#include <tensorrt_llm/common/tllmException.h>

namespace rust::behavior {
    template<typename Try, typename Fail>
    static void trycatch(Try &&func, Fail &&fail) noexcept try {
        func();
    } catch (tensorrt_llm::common::TllmException &e) {
        fail(e.what());
    }
}

#include "backends/trtllm/src/lib.rs.h"

namespace huggingface::tgi::backends {

    class TensorRtLlmBackendImpl : public TensorRtLlmBackend {
    public:
        /***
         *
         * @param engineFolder
         * @param executorWorker
         */
        TensorRtLlmBackendImpl(const std::string_view &engineFolder, const std::string_view &executorWorker);

        /***
         *
         * @param tokens
         * @param maxNewTokens
         * @param topK
         * @param topP
         * @param temperature
         * @param repetition_penalty
         * @param frequency_penalty
         * @param seed
         * @return
         */
        [[nodiscard("returned request id should be used to refer to the request's generation result later on")]]
        uint64_t
        Submit(rust::Slice<const uint32_t> tokens, uint32_t maxNewTokens,
               int32_t topK, float_t topP, float_t temperature,
               float_t repetition_penalty, float_t frequency_penalty, uint64_t seed);

        /***
         *
         * @return
         */
        std::unique_ptr<std::vector<GenerationStep>> PullTokens();
    };

    /***
    *
    * @param engineFolder
    * @return
    */
    std::unique_ptr<TensorRtLlmBackendImpl> CreateTensorRtLlmBackend(rust::Str engineFolder, rust::Str executorWorker);
}

#endif //TGI_TRTLLM_BACKEND_FFI_H
