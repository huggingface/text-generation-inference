//
// Created by mfuntowicz on 7/11/24.
//

#ifndef TGI_TRTLLM_BACKEND_FFI_H
#define TGI_TRTLLM_BACKEND_FFI_H

//#include "rust/cxx.h"
#include "backend.h"

namespace huggingface::tgi::backends {
    class TensorRtLlmBackendImpl;
}

#include "backends/trtllm/src/lib.rs.h"


namespace huggingface::tgi::backends {

    struct GenerationContext;

    class TensorRtLlmBackendImpl : TensorRtLlmBackend {
    public:
        /***
         *
         * @param engineFolder
         * @param executorWorker
         */
        TensorRtLlmBackendImpl(const std::string_view &engineFolder, const std::string_view &executorWorker);

        /***
         *
         * @return
         */
        bool IsReady() const;

        /***
         *
         * @param tokens
         * @param maxNewTokens
         * @param topK
         * @param topP
         * @param temperature
         * @param seed
         * @return
         */
        [[nodiscard("returned request id should be used to refer to the request's generation result later on")]]
        uint64_t Submit(rust::Slice<const uint32_t> tokens, int32_t maxNewTokens, int32_t topK, float_t topP, float_t temperature, uint64_t seed);

        /***
         *
         * @param requestId
         * @param handler
         * @return
         */
        uint32_t Stream(rust::Box <GenerationContext> ctx,
                        uint64_t requestId,
                        rust::Fn<void(rust::Box<GenerationContext>, uint32_t, uint32_t, bool)> handler);
    };

    /***
    *
    * @param engineFolder
    * @return
    */
    std::unique_ptr<TensorRtLlmBackendImpl> CreateTensorRtLlmBackend(rust::Str engineFolder, rust::Str executorWorker);
}

#endif //TGI_TRTLLM_BACKEND_FFI_H
