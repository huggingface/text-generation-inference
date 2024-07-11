//
// Created by Morgan Funtowicz on 6/30/24.
//

#ifndef TGI_TRTLLM_BACKEND_H
#define TGI_TRTLLM_BACKEND_H

#include <cmath>
#include <filesystem>
#include <span>
#include <vector>

#include <spdlog/fmt/fmt.h>
#include <nlohmann/json.hpp>

#include <tensorrt_llm/runtime/common.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

using json = nlohmann::json;
namespace tle = tensorrt_llm::executor;

namespace huggingface::tgi::backends {
    using RequestId = tle::IdType;
    using TokenId = tle::TokenIdType;
    using TokenStreamingCallback = void(tle::TokenIdType);

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

        /***
         * Indicate if the backend is ready to accept incoming request
         * @return true if ready, false otherwise
         */
        [[nodiscard]] bool IsReady() const;

        /***
         * Submit a new generation task to the executor
         * @param tokens
         * @param maxNewTokens
         * @param topK
         * @param topP
         * @param temperature
         * @param seed
         * @return Request id related to this generation for reference
         */
        [[nodiscard]] RequestId Submit(
                const std::vector<TokenId> &tokens,
                int32_t maxNewTokens,
                int32_t topK,
                float_t topP,
                float_t temperature,
                uint64_t seed
        );

        /***
         *
         * @param requestId The request id to poll the generation results
         * @return
         */
        std::vector<tle::Response> Poll(RequestId requestId);

        /***
         * Unroll the token generation until end of stream is reached.
         * Every generated token is streamed back through the provided callback for further processing
         * @param reqId The request id to unroll
         * @param cb The callback to stream token back
         * @return Global number of generated tokens for this request id
         */
        uint32_t Stream(RequestId reqId, std::function<TokenStreamingCallback> &cb);
    };
}


#endif //TGI_TRTLLM_BACKEND_H