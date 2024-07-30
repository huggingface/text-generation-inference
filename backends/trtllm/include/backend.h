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
            uint32_t topK,
            float_t topP,
            float_t temperature,
            float_t repetition_penalty,
            float_t frequency_penalty,
            uint64_t seed
    );

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
         * Indicate if the backend is ready to accept incoming request
         * @return true if ready, false otherwise
         */
        [[nodiscard]] bool IsReady() const;

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
                int32_t topK,
                float_t topP,
                float_t temperature,
                float_t repetition_penalty,
                float_t frequency_penalty,
                uint64_t seed
        );

        /**
         *
         * @param requestId The request id to poll the generation results
         * @return
         */
        std::vector<tle::Response> Poll(RequestId requestId);

        /**
         * Stop the underlying executor
         */
        void Shutdown();
    };
}


#endif //TGI_TRTLLM_BACKEND_H
