//
// Created by Morgan Funtowicz on 6/30/24.
//

#ifndef TGI_TRTLLM_BACKEND_H
#define TGI_TRTLLM_BACKEND_H

#include <cmath>
#include <filesystem>
#include <span>
#include <vector>

#include <nlohmann/json.hpp>

#include <tensorrt_llm/runtime/common.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

using json = nlohmann::json;
namespace tle = tensorrt_llm::executor;

namespace huggingface::tgi::backends {
    using RequestId = tle::IdType;
    using TokenId = tle::TokenIdType;

    const static auto OUTPUT_CONFIG = tle::OutputConfig(true, false, false, true, false);

    /**
     * Initialize all the components required by TRTLLM.
     * It is required to call this function before attempting to load any engine
     */
    void InitializeBackend();

    /**
     *
     * @param config TensorRT-LLM configuration object
     * @param workerPath Path to the "executorWorker" provided by TensorRT-LLM when using orchestrator mode
     * @return
     */
    tle::ExecutorConfig GetExecutorConfig(const json &config, const std::string &workerPath);

    /**
     * Get the sampling configuration from the parameters provided by TGI
     * @param topK
     * @param topP
     * @param temperature
     * @param repetition_penalty
     * @param frequency_penalty
     * @param seed
     * @return
     */
    tle::SamplingConfig GetSamplingConfig(
            const uint32_t topK,
            const float_t topP,
            const float_t temperature,
            const float_t repetition_penalty,
            const float_t frequency_penalty,
            const uint64_t seed
    ) noexcept;

    /**
     *
     */
    class TensorRtLlmBackend {
    private:
        const json config;
        tle::Executor executor;

    public:
        explicit TensorRtLlmBackend(
                const std::filesystem::path &engineFolder,
                const std::filesystem::path &executorWorker
        );

        /**
         * Query the executor for the number of token available for pulling
         * @return
         */
        [[nodiscard]] size_t NumResponsesReady() const;

        /**
         * Submit a new generation task to the executor
         * @param tokens
         * @param topK
         * @param topP
         * @param temperature
         * @param repetition_penalty
         * @param frequency_penalty
         * @param seed
         * @return Request id related to this generation for reference
         */
        [[nodiscard]] RequestId Submit(
                const std::vector<TokenId> &tokens,
                const uint32_t maxNewTokens,
                const int32_t topK,
                const float_t topP,
                const float_t temperature,
                const float_t repetition_penalty,
                const float_t frequency_penalty,
                const uint64_t seed
        );

        [[nodiscard]] std::vector<tle::Response> PullNewTokens();
    };
}


#endif //TGI_TRTLLM_BACKEND_H
