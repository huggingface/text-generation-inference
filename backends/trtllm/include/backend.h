//
// Created by Morgan Funtowicz on 6/30/24.
//

#ifndef TGI_TRTLLM_BACKEND_H
#define TGI_TRTLLM_BACKEND_H

#include <filesystem>
#include <span>

#include <nlohmann/json.hpp>
#include <fmt/format.h>

#include <tensorrt_llm/runtime/common.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

using json = nlohmann::json;
namespace tle = tensorrt_llm::executor;

namespace huggingface::tgi::backends {

    tle::ExecutorConfig GetExecutorConfig(const json &config, const std::string &workerPath);

    class TensorRtLlmBackend {
    private:
        const json config;
        tle::Executor executor;

    public:
        explicit TensorRtLlmBackend(
                const std::filesystem::path &engineFolder,
                const std::filesystem::path &executorWorker
        );

        /***
         * Indicate if the backend is ready to accept incoming request
         * @return true if ready, false otherwise
         */
        [[nodiscard]] bool IsReady() const {
            return executor.canEnqueueRequests();
        }

        /***
         *
         * @param tokens
         * @param maxNewTokens
         * @param topK
         * @param topP
         * @param temperature
         * @param minLength
         * @param repetitionPenalty
         * @param frequencePenalty
         * @param seed
         * @param nTopTokens
         * @return
         */
        [[nodiscard]] tle::IdType Submit(
                std::vector<tle::TokenIdType> &tokens,
                int32_t maxNewTokens,
                float_t topK,
                float_t topP,
                float_t temperature,
                int32_t minLength,
                std::optional<float_t> repetitionPenalty = std::nullopt,
                std::optional<float_t> frequencePenalty = std::nullopt,
                std::optional<uint32_t> seed = std::nullopt,
                std::optional<uint32_t> nTopTokens = std::nullopt
        );
    };
}


#endif //TGI_TRTLLM_BACKEND_H
