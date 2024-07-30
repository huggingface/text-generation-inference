#include <fstream>

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <nvml.h>

#include "backend.h"
#include "hardware.h"

void huggingface::tgi::backends::InitializeBackend() {
    SPDLOG_INFO("Initializing Backend...");
    nvmlInit_v2();
    initTrtLlmPlugins();

    const auto numGpus = huggingface::hardware::cuda::GetNumDevices();
    if (numGpus.has_value()) {
        SPDLOG_INFO("Detected {:d} Nvidia GPU(s)", numGpus.value());
    } else {
        SPDLOG_WARN("Failed to detected Nvidia GPU(s) on the system");
    }
}

[[nodiscard]]
tle::ExecutorConfig huggingface::tgi::backends::GetExecutorConfig(const json &config, const std::string &workerPath) {
    tle::ExecutorConfig execConfig(1);

    // Retrieve the compute capabilities to enable some options at runtime
    const auto computeCapabilities = huggingface::hardware::cuda::GetCudaComputeCapabilities();

    // Single engine (TP = PP = 1) -> using leader mode (no MPI involved)
    if (config["/pretrained_config/mapping/world_size"_json_pointer].get<uint8_t>() == 1) {
        SPDLOG_INFO("Detected single engine deployment, using leader mode");
        execConfig.setParallelConfig(tle::ParallelConfig(
                tle::CommunicationType::kMPI,
                tle::CommunicationMode::kLEADER,
                std::nullopt,
                std::nullopt,
                std::nullopt
        ));
    } else { // Multiple engines -> using orchestrator mode (MPI involved)
        SPDLOG_INFO("Detected sharded engine deployment, using orchestrator mode");
        execConfig.setParallelConfig(tle::ParallelConfig(
                tle::CommunicationType::kMPI,
                tle::CommunicationMode::kORCHESTRATOR,
                std::nullopt,
                std::nullopt,
                tle::OrchestratorConfig(true, workerPath, nullptr, true)
        ));
    }

    // Define some configuration variables
    execConfig.setKvCacheConfig(tle::KvCacheConfig(true));
    execConfig.setEnableChunkedContext(computeCapabilities.isPostAmpere());
    return execConfig;
}

tle::SamplingConfig huggingface::tgi::backends::GetSamplingConfig(
        uint32_t topK,
        float_t topP,
        float_t temperature,
        float_t repetition_penalty,
        float_t frequency_penalty,
        uint64_t seed) {
    return tle::SamplingConfig(
            1,  // TGI only use a single beam
            topK,
            topP,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            seed,
            temperature,
            temperature,
            std::nullopt,
            repetition_penalty,
            std::nullopt,
            frequency_penalty
    );
}

huggingface::tgi::backends::TensorRtLlmBackend::TensorRtLlmBackend(
        const std::filesystem::path &enginesFolder,
        const std::filesystem::path &executorWorker
) :
        config(json::parse(std::ifstream(enginesFolder / "config.json"))),
        executor(
                enginesFolder,
                tensorrt_llm::executor::ModelType::kDECODER_ONLY,
                GetExecutorConfig(config, executorWorker.string()
                )) {
    SPDLOG_INFO(FMT_STRING("Engine (version={})"), config["/version"_json_pointer].get_ref<const std::string &>());
}

bool huggingface::tgi::backends::TensorRtLlmBackend::IsReady() const {
    return executor.canEnqueueRequests();
}

[[nodiscard("Returned number of requests needs to be consumed")]]
size_t huggingface::tgi::backends::TensorRtLlmBackend::NumResponsesReady() const {
    return executor.getNumResponsesReady();
}

[[nodiscard("Returned request id needs to be provided back to gather generated tokens")]]
tle::IdType huggingface::tgi::backends::TensorRtLlmBackend::Submit(
        const std::vector<tle::TokenIdType> &tokens,
        const int32_t topK,
        const float_t topP,
        const float_t temperature,
        const float_t repetition_penalty,
        const float_t frequency_penalty,
        const uint64_t seed
) {
#ifdef NDEBUG
    SPDLOG_DEBUG(
            FMT_STRING("Submitting inference over {:d} tokens to the executor ({:d} already in-flight)"),
            tokens.size(),
            executor.getLatestIterationStats().back().numActiveRequests
    );
#else
    SPDLOG_DEBUG(
            FMT_STRING("Submitting inference [{}] to the executor ({:d} already in-flight)"),
            fmt::join(tokens, ", "),
            executor.getLatestIterationStats().front().numActiveRequests
    );
#endif

    const auto maxNumTokens = config["/build_config/max_num_tokens"_json_pointer].get<size_t>();
    const auto maxNewTokens = static_cast<int32_t>(std::max(1ul, maxNumTokens - tokens.size()));

    const auto sampling = GetSamplingConfig(topK, topP, temperature, repetition_penalty, frequency_penalty, seed);
    const auto output = tle::OutputConfig(true, false, false, true, false);
    return executor.enqueueRequest(
            tle::Request{tokens, maxNewTokens, true, sampling, output});
}

[[nodiscard("Generated tokens result must be used")]]
std::vector<tle::Response> huggingface::tgi::backends::TensorRtLlmBackend::Poll(const tle::IdType requestId) {
    SPDLOG_DEBUG(FMT_STRING("Polling status for request {:d}"), requestId);
    return executor.awaitResponses(requestId);
}


void huggingface::tgi::backends::TensorRtLlmBackend::Shutdown() {
    SPDLOG_INFO("Shutting down executor");
    executor.shutdown();
}
