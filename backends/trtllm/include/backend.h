//
// Created by Morgan Funtowicz on 6/30/24.
//

#ifndef TGI_TRTLLM_BACKEND_H
#define TGI_TRTLLM_BACKEND_H

#include <array>
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


#define CAST_SIZETYPE(x) static_cast<tle::SizeType32>(x)

namespace huggingface::tgi::backends {
    using RequestId = tle::IdType;
    using TokenId = tle::TokenIdType;

    const static auto OUTPUT_CONFIG = tle::OutputConfig(true, false, false, true, false);
    constexpr auto FMT_NOT_ENOUGH_GPUS = FMT_STRING(
            "Not enough GPUs to allocate requested model (detected: {:d}, required: {:d})");
    constexpr auto FMT_EXECUTOR_STATS = FMT_STRING(
            "Submitting inference [{}] to the executor ({:d} already in-flight)");
    constexpr auto FMT_SAMPLING_CONFIG = FMT_STRING(
            "Sampling: topK={:d}, topP={:.1f}, temperature={:.1f}, repetition_penalty={:.1f}, frequency_penalty={:.1f}, seed={:d}");

    /**
     * Initialize all the components required by TRTLLM.
     * It is required to call this function before attempting to load any engine
     */
    void InitializeBackend();

    /**
     * Initialize logging mechanism
     */
    void InitializeLogging();


    /**
     *
     * @param config TensorRT-LLM configuration object
     * @param workerPath Path to the "executorWorker" provided by TensorRT-LLM when using orchestrator mode
     * @return
     */
    tle::ExecutorConfig GetExecutorConfig(const json &config, const std::string &workerPath);

    /**
     *
     * @param worldSize
     * @param workerPath
     * @return
     */
    tle::ParallelConfig GetParallelConfig(size_t worldSize, std::string workerPath) noexcept;

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
    ) noexcept;

    /**
     * Attempt to retrieve the
     * @param generationConfigPath
     * @return
     */
    std::optional<std::list<std::vector<TokenId>>>
    GetStopWordsFromConfig(const std::filesystem::path &generationConfigPath) noexcept;

    /**
     *
     */
    class TensorRtLlmBackend {
    private:
        const json config;
        tle::Executor executor;

        /** Frequently accessed variables cached here **/
        uint32_t maxNumTokens;
        std::list<std::vector<TokenId>> stopWords;

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
         * @param repetitionPenalty
         * @param frequencyPenalty
         * @param seed
         * @return Request id related to this generation for reference
         */
        [[nodiscard]] RequestId Submit(
                const std::vector<TokenId> &tokens,
                uint32_t maxNewTokens,
                int32_t topK,
                float_t topP,
                float_t temperature,
                float_t repetitionPenalty,
                float_t frequencyPenalty,
                uint64_t seed
        );

        [[nodiscard]] std::vector<tle::Response> PullNewTokens();
    };
}


#endif //TGI_TRTLLM_BACKEND_H
