#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "backend.h"

void huggingface::tgi::backends::InitializeBackend() {
    SPDLOG_INFO("Initializing Backend...");

    initTrtLlmPlugins();
}

tle::ExecutorConfig huggingface::tgi::backends::GetExecutorConfig(const json &config, const std::string &workerPath) {
    tle::ExecutorConfig execConfig(1);

    // TODO : Need to check for >= sm_80 (ampere)
    // execConfig.setEnableChunkedContext(true)
    execConfig.setKvCacheConfig(tle::KvCacheConfig(true));

    if(config["/pretrained_config/mapping/world_size"_json_pointer].get<uint8_t>() == 1){
        SPDLOG_INFO("Detected single engine deployment, using leader mode");
        execConfig.setParallelConfig(tle::ParallelConfig(
                tle::CommunicationType::kMPI,
                tle::CommunicationMode::kLEADER,
                std::nullopt,
                std::nullopt,
                std::nullopt
        ));
    } else {
        SPDLOG_INFO("Detected sharded engine deployment, using orchestrator mode");
        execConfig.setParallelConfig(tle::ParallelConfig(
                tle::CommunicationType::kMPI,
                tle::CommunicationMode::kORCHESTRATOR,
                std::nullopt,
                std::nullopt,
                tle::OrchestratorConfig(true, workerPath)
        ));
    }
    return execConfig;
}

huggingface::tgi::backends::TensorRtLlmBackend::TensorRtLlmBackend(
        const std::filesystem::path &enginesFolder,
        const std::filesystem::path &executorWorker
):
    config(json::parse(std::ifstream(enginesFolder / "config.json"))),
    executor(
        enginesFolder,
        tensorrt_llm::executor::ModelType::kDECODER_ONLY,
        GetExecutorConfig(config, executorWorker.string()
    ))
{
    SPDLOG_INFO(FMT_STRING("Engine (version={})"), config["/version"_json_pointer].get_ref<const std::string&>());
}

tle::IdType huggingface::tgi::backends::TensorRtLlmBackend::Submit(
        const std::vector<tle::TokenIdType> &tokens,
        const int32_t maxNewTokens,
        const int32_t topK,
        const float_t topP,
        const float_t temperature,
        const int32_t minLength,
        std::optional<float_t> repetitionPenalty,
        std::optional<float_t> frequencyPenalty,
        std::optional<uint32_t> seed,
        std::optional<uint32_t> nTopTokens
) {
    spdlog::debug(
            "Submitting inference over {:d} tokens to the executor {:d}",
            tokens.size(),
            executor.getLatestIterationStats().back().numActiveRequests
    );

    const auto sampling = tle::SamplingConfig{
            1,
            topK,
            topP,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            seed,
            temperature,
            minLength,
            std::nullopt,
            repetitionPenalty,
            std::nullopt,
            frequencyPenalty,
    };
    const auto output = tle::OutputConfig{false, false, nTopTokens.value_or(1) > 1};
    const auto request = tle::Request{tokens, maxNewTokens, true, sampling, output};

    return executor.enqueueRequest(request);
}

std::vector<tle::Response> huggingface::tgi::backends::TensorRtLlmBackend::Poll(const tle::IdType reqId) {
    SPDLOG_DEBUG("Polling request {:d}", reqId);
    const auto responses = executor.awaitResponses(reqId);
    return responses;
}