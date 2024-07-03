#include <spdlog/spdlog.h>
#include <fmt/std.h>

#include "backend.h"

huggingface::tgi::backends::TensorRtLlmBackend::TensorRtLlmBackend(const std::filesystem::path &engineFolder)
        : executor(engineFolder, tle::ModelType::kDECODER_ONLY, tle::ExecutorConfig{}) {
    SPDLOG_INFO(FMT_STRING("Loading engines from {}"), engineFolder);
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
    if (IsReady()) {
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
                repetitionPenalty.value_or(0.0),
                std::nullopt,
                frequencePenalty.value_or(1.0),
        };
        const auto output = tle::OutputConfig{false, false, nTopTokens.value_or(1) > 1};
        const auto request = tle::Request{std::move(tokens), maxNewTokens, true, sampling, output};

        return executor.enqueueRequest(request);
    }
    return 0;
}
