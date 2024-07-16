#include <fstream>

#include <nvml.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "backend.h"

void huggingface::tgi::backends::InitializeBackend() {
    SPDLOG_INFO("Initializing Backend...");
    nvmlInit_v2();
    initTrtLlmPlugins();
}

[[nodiscard]]
tle::ExecutorConfig huggingface::tgi::backends::GetExecutorConfig(const json &config, const std::string &workerPath) {
    tle::ExecutorConfig execConfig(1);

    // Get the compute capabilities of the current hardware
    nvmlDevice_t device;
    int32_t cudaComputeCapabilitiesMajor = 0, cudaComputeCapabilitiesMinor = 0;
    if (nvmlDeviceGetHandleByIndex_v2(0, &device) == NVML_SUCCESS) {
        SPDLOG_DEBUG("Successfully acquired nvmlDevice_t = 0");
        if (nvmlDeviceGetCudaComputeCapability(device, &cudaComputeCapabilitiesMajor, &cudaComputeCapabilitiesMinor) ==
            NVML_SUCCESS) {
            SPDLOG_INFO(FMT_STRING("Detected sm_{:d}{:d} compute capabilities"), cudaComputeCapabilitiesMajor,
                        cudaComputeCapabilitiesMinor);
        }
    }

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
                tle::OrchestratorConfig(true, workerPath)
        ));
    }

    // Define some configuration variables
    execConfig.setKvCacheConfig(tle::KvCacheConfig(true));
    execConfig.setEnableChunkedContext(cudaComputeCapabilitiesMajor >= 8);
    return execConfig;
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

size_t huggingface::tgi::backends::TensorRtLlmBackend::NumResponsesReady() const {
    return executor.getNumResponsesReady();
}

[[nodiscard("Returned request id needs to be provided back to gather generated tokens")]]
tle::IdType huggingface::tgi::backends::TensorRtLlmBackend::Submit(
        const std::vector<tle::TokenIdType> &tokens,
        const int32_t topK,
        const float_t topP,
        const float_t temperature,
        const uint64_t seed
) {
#ifndef NDEBUG
    SPDLOG_INFO(
            FMT_STRING("Submitting inference over {:d} tokens to the executor ({:d} already in-flight)"),
            tokens.size(),
            executor.getLatestIterationStats().back().numActiveRequests
    );
#else
    SPDLOG_INFO(
            FMT_STRING("Submitting inference [{}] to the executor ({:d} already in-flight)"),
            fmt::join(tokens, ", "),
            executor.getLatestIterationStats().back().numActiveRequests
    );
#endif

    const auto sampling = tle::SamplingConfig{
            1,
            topK,
            topP,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            seed,
            std::nullopt,
            temperature,
            std::nullopt,
    };
    const auto output = tle::OutputConfig{false, false, false};
    return executor.enqueueRequest(
            tle::Request{tokens, std::numeric_limits<tle::SizeType32>::max(), true, sampling, output});
}

[[nodiscard("Generated tokens result must be used")]]
std::vector<tle::Response> huggingface::tgi::backends::TensorRtLlmBackend::Poll(const tle::IdType requestId) {
    SPDLOG_INFO("Polling status for request {}", requestId);
    return executor.awaitResponses(requestId);
}


void huggingface::tgi::backends::TensorRtLlmBackend::Shutdown() {
    SPDLOG_INFO("Shutting down executor");
    executor.shutdown();
}