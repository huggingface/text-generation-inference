//
// Created by mfuntowicz on 7/11/24.
//

#ifndef TGI_TRTLLM_BACKEND_FFI_H
#define TGI_TRTLLM_BACKEND_FFI_H

#include <cstddef>
#include "backend.h"

namespace huggingface::tgi::backends {
    class TensorRtLlmBackendImpl;
}

#include "backends/trtllm/src/lib.rs.h"


namespace huggingface::tgi::backends {

//    struct GenerationContext;

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
         * @return
         */
        bool IsReady() const;

        /***
         *
         * @param tokens
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
        Submit(rust::Slice<const uint32_t> tokens, int32_t topK, float_t topP, float_t temperature,
               float_t repetition_penalty, float_t frequency_penalty, uint64_t seed);

        /***
         *
         * @param requestId
         * @param ctx
         * @param callback
         * @return
         */
        size_t StreamTokens(
                const RequestId requestId,
                huggingface::tgi::backends::GenerationContext *ctx,
                rust::Fn<void(huggingface::tgi::backends::GenerationContext *,
                              huggingface::tgi::backends::GenerationStep)> callback);
    };

    /***
    *
    * @param engineFolder
    * @return
    */
    std::unique_ptr<TensorRtLlmBackendImpl> CreateTensorRtLlmBackend(rust::Str engineFolder, rust::Str executorWorker);
}

#endif //TGI_TRTLLM_BACKEND_FFI_H
