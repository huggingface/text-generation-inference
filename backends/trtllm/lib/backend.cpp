#include <spdlog/spdlog.h>
#include <fmt/std.h>

#include "backend.h"

tle::ExecutorConfig huggingface::tgi::backends::GetExecutorConfig(const json &config, const std::string &workerPath) {
    tle::ExecutorConfig execConfig(
            config["/build_config/max_beam_width"_json_pointer].get<int32_t>()
    );

    execConfig.setParallelConfig(tle::ParallelConfig(
            tle::CommunicationType::kMPI,
            tle::CommunicationMode::kORCHESTRATOR,
            std::nullopt,
            std::nullopt,
            tle::OrchestratorConfig(true, workerPath)
    ));


    // TODO : Need to check for >= sm_80 (ampere)
    // execConfig.setEnableChunkedContext(true)
    execConfig.setKvCacheConfig(tle::KvCacheConfig(true));
    return execConfig;
}

huggingface::tgi::backends::TensorRtLlmBackend::TensorRtLlmBackend(
        const std::filesystem::path &engineFolder,
        const std::filesystem::path &executorWorker
):
    config(json::parse(std::ifstream(engineFolder / "config.json"))),
    executor(engineFolder, tensorrt_llm::executor::ModelType::kDECODER_ONLY, GetExecutorConfig(config, executorWorker.string()))
{
    initTrtLlmPlugins();
    SPDLOG_INFO(FMT_STRING("Engine (version={})"), config["version"].get<std::string>());
}

tle::IdType huggingface::tgi::backends::TensorRtLlmBackend::Submit(
        std::vector<tle::TokenIdType> &tokens,
        const int32_t maxNewTokens,
        const float_t topK,
        const float_t topP,
        const float_t temperature,
        const int32_t minLength,
        const std::optional<float_t> repetitionPenalty,
        const std::optional<float_t> frequencePenalty,
        const std::optional<uint32_t> seed,
        const std::optional<uint32_t> nTopTokens
) {
//    if (IsReady()) {
//        spdlog::debug(
//                "Submitting inference over {:d} tokens to the executor {:d}",
//                tokens.size(),
//                executor.getLatestIterationStats().back().numActiveRequests
//        );
//
//        const auto sampling = tle::SamplingConfig{
//                1,
//                topK,
//                topP,
//                std::nullopt,
//                std::nullopt,
//                std::nullopt,
//                seed,
//                temperature,
//                minLength,
//                std::nullopt,
//                repetitionPenalty.value_or(0.0),
//                std::nullopt,
//                frequencePenalty.value_or(1.0),
//        };
//        const auto output = tle::OutputConfig{false, false, nTopTokens.value_or(1) > 1};
//        const auto request = tle::Request{std::move(tokens), maxNewTokens, true, sampling, output};
//
//        return executor.enqueueRequest(request);
//    }
    return 0;
}
