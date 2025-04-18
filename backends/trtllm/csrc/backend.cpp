#include <ranges>

#include <nlohmann/json.hpp>

#include "backend.hpp"
#include "hardware.hpp"

namespace huggingface::tgi::backends::trtllm {
    tle::ParallelConfig backend_workspace_t::parallel_config() const {
        // Single engine (TP = PP = 1) -> using leader mode (no MPI involved)
        const auto world_size = config_["/pretrained_config/mapping/world_size"_json_pointer].get<size_t>();

        auto mode = tle::CommunicationMode::kLEADER;
        std::optional<tle::OrchestratorConfig> orchestratorConfig = std::nullopt;

        if (world_size > 1) {
            SPDLOG_INFO("Detected sharded engine deployment, using orchestrator mode");
            mode = tle::CommunicationMode::kORCHESTRATOR;
            orchestratorConfig = std::make_optional<tle::OrchestratorConfig>(true, executor_worker_path_, nullptr,
                                                                             true);
        } else {
            SPDLOG_INFO("Detected single engine deployment, using leader mode");
        }

        return tle::ParallelConfig(tle::CommunicationType::kMPI, mode, std::nullopt, std::nullopt, orchestratorConfig);
    }


    tle::ExecutorConfig backend_workspace_t::executor_config() const {
        // Retrieve the compute capabilities to enable some options at runtime
        const auto compute_capabilities = hardware::cuda::compute_capabilities_t();

        // Allocate the config
        tle::ExecutorConfig executor_config(/* maxBeamWidth = */ 1);

        // Set the parallel config as inferred
        executor_config.setParallelConfig(parallel_config());

        // Define some configuration variables
        executor_config.setKvCacheConfig(tle::KvCacheConfig(true));
        executor_config.setEnableChunkedContext(compute_capabilities.is_at_least_ampere());
        executor_config.setSchedulerConfig(tle::SchedulerConfig(tle::CapacitySchedulerPolicy::kMAX_UTILIZATION));
        return executor_config;
    }

    backend_t::backend_t(std::filesystem::path &engines_folder, std::filesystem::path &executor_worker_path)
            : workspace(engines_folder, executor_worker_path), executor_(executor_factory_initializer(workspace)) {}

    size_t backend_t::num_tokens_ready() const noexcept {
        return executor_.getNumResponsesReady();
    }

    std::expected<request_id_t, backend_error_t>
    backend_t::submit(std::span<const token_id_t> token_ids, const generation_params_t g_params,
                      const sampling_params_t s_params) noexcept {
        SPDLOG_DEBUG("Submit {:d} tokens for scheduling ({}, {})", token_ids.size(), g_params, s_params);
        return executor_.enqueueRequest(tle::Request{
                {token_ids.begin(), token_ids.end()},  // Making actual copy of the tokens
                static_cast<tle::SizeType32>(g_params.max_new_tokens),
                true,
                (tle::SamplingConfig) s_params,
                tle::OutputConfig{ /* returnLogProbs= */ true},
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                workspace.generation_config().stop_words
        });
    }

    std::vector<tle::Response> backend_t::pull_tokens() noexcept {
        SPDLOG_TRACE(FMT_STRING("Pulling out tokens ({:d} available)"), num_tokens_ready());
        return executor_.awaitResponses();
    }

    void backend_t::cancel(request_id_t request_id) noexcept {
        SPDLOG_TRACE(FMT_STRING("Cancelling request: {:d}"), request_id);
        executor_.cancelRequest(request_id);
    }
}
