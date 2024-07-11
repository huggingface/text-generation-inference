#include <fstream>

#include <nvml.h>
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

[[nodiscard("Returned request id needs to be provided back to gather generated tokens")]]
tle::IdType huggingface::tgi::backends::TensorRtLlmBackend::Submit(
        const std::vector<tle::TokenIdType> &tokens,
        const int32_t maxNewTokens,
        const int32_t topK,
        const float_t topP,
        const float_t temperature,
        const uint64_t seed
) {
    SPDLOG_DEBUG(
            FMT_STRING("Submitting inference over {:d} tokens to the executor ({:d} already in-flight)"),
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
            std::nullopt,
            temperature,
            std::nullopt,
    };
    const auto output = tle::OutputConfig{false, false, false};
    return executor.enqueueRequest(tle::Request{tokens, maxNewTokens, true, sampling, output});
}

uint32_t huggingface::tgi::backends::TensorRtLlmBackend::Stream(const tle::IdType reqId,
                                                                std::function<TokenStreamingCallback> &cb) {
    bool isFinal = false;
    size_t generatedTokens = 0;

    do {
        const auto responses = executor.awaitResponses(reqId);
        for (const auto &response: responses) {
            if (response.hasError()) {
                SPDLOG_WARN("Caught error during generation: {}", response.getErrorMsg());
                isFinal = true;
            } else {
                const auto generation = response.getResult();
                const auto token = generation.outputTokenIds[0][0];

                // Update the end of stream detection and overall number of generated tokens
                isFinal = generation.isFinal;
                ++generatedTokens;

                // Send the token back through the callback function for further processing
                cb(token);
            }
        }

    } while (!isFinal);

    // Return the number of generated tokens
    return generatedTokens;
}

std::vector<tle::Response> huggingface::tgi::backends::TensorRtLlmBackend::Poll(const tle::IdType requestId) {
    return executor.awaitResponses(requestId);
}
