#ifndef TGI_BACKEND_TRTLLM_FFI
#define TGI_BACKEND_TRTLLM_FFI

#include <memory>
#include <thread>

#include <nvml.h>
#include <tensorrt_llm/common/tllmException.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

#include <spdlog/spdlog.h>

#include <backend.hpp>
#include <hardware.hpp>

namespace rust::behavior {
    template<typename Try, typename Fail>
    static void trycatch(Try &&func, Fail &&fail) noexcept try {
        func();
    } catch (tensorrt_llm::common::TllmException &e) {
        fail(e.what());
    }
}

namespace huggingface::tgi::backends::trtllm {
    class tensorrt_llm_backend_t;
}

#include "backends/trtllm/src/lib.rs.h"

namespace huggingface::tgi::backends::trtllm {
    std::once_flag backend_initialized_flag;

    class tensorrt_llm_backend_t {
    private:
        backend_t inner_;

    public:
        tensorrt_llm_backend_t(std::filesystem::path &&engine_folder, std::filesystem::path &&executor_worker_path)
            : inner_(engine_folder, executor_worker_path) {}

        size_t num_tokens_ready() const noexcept {
            return inner_.num_tokens_ready();
        }

        request_id_t submit(
                rust::Slice<const uint32_t> tokens,
                uint32_t max_new_tokens,
                uint32_t top_k,
                float_t top_p,
                float_t temperature,
                float_t repetition_penalty,
                float_t frequency_penalty,
                uint64_t seed
        ) {
            // This is enabled only if using add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
            SPDLOG_TRACE(FMT_STRING("[FFI] Submitting {:d} prompt tokens to the executor"));

            // Submit the request to the executor and get back a potential request_id used to track request status
            const auto signed_tokens = std::vector<int32_t>(tokens.begin(), tokens.end());
            const auto maybe_request_id = inner_.submit(
                signed_tokens,
                {max_new_tokens},
                {top_k, top_p, repetition_penalty, frequency_penalty, temperature, seed}
            );

            // If we do have a value, let's return the request_id
            if(maybe_request_id.has_value()) [[likely]] {
                return *maybe_request_id;
            } else {
                SPDLOG_WARN("[FFI] Failed to submit request to the executor");
                return maybe_request_id.error();
            }
        }

        std::unique_ptr<std::vector<generation_step_t>> pull_tokens() noexcept {
            if(num_tokens_ready() > 0) [[likely]] {
                const auto responses = inner_.pull_tokens();

                SPDLOG_TRACE("[FFI] Successfully pulled out {:d} responses from executor", responses.size());
                // Transform tle::Response to GenerationStep
                auto steps = std::make_unique<std::vector<generation_step_t>>();
                std::ranges::transform(responses.begin(), responses.end(), std::back_inserter(*steps), [](const tle::Response &r) {
                    const auto reqId = r.getRequestId();
                    if (!r.hasError()) [[likely]] {
                        const auto result = r.getResult();
                        return generation_step_t{
                                reqId,
                                static_cast<uint32_t>(result.outputTokenIds[0][0]),
                                result.logProbs.value()[0][0],
                                result.isFinal,
                                false,
                                std::string()
                        };
                    } else {
                        return generation_step_t{
                                reqId,
                                0,
                                0.0,
                                true,
                                true,
                                std::move(r.getErrorMsg())
                        };
                    }
                });
                return steps;

            } else {
                return std::make_unique<std::vector<generation_step_t>>();
            }
        }

        void cancel(request_id_t requestId) noexcept {
            SPDLOG_DEBUG("[FFI] cancelling request {:d}", requestId);
            inner_.cancel(requestId);
        }
    };

    void initialize_logging() {
#ifndef TGI_TRTLLM_BACKEND_DEBUG
        if (const auto TRTLLM_LOG_LEVEL_CSTR = std::getenv("TRTLLM_LOG_LEVEL")) {
        std::string log_level(TRTLLM_LOG_LEVEL_CSTR);
        std::transform(log_level.begin(), log_level.end(), log_level.begin(), [](unsigned char c) {
            return std::tolower(c);
        });

        if (log_level == "debug")
            spdlog::set_level(spdlog::level::debug);
        else
            spdlog::set_level(spdlog::level::info);
    }
#else
        spdlog::set_level(spdlog::level::debug);
#endif
    }

    void initialize_tensorrt_llm_backend() {
        SPDLOG_INFO("Initializing TGI - TensoRT-LLM Backend (v{})", tle::version());

        // Initialize everyone
        initialize_logging();
        nvmlInit_v2();
        initTrtLlmPlugins();

        const auto numGpus = huggingface::tgi::hardware::cuda::get_device_count();
        if (numGpus.has_value()) {
            SPDLOG_INFO("[FFI] Detected {:d} Nvidia GPU(s)", *numGpus);
        } else {
            SPDLOG_WARN("[FFI] Failed to detected Nvidia GPU(s) on the system");
            // todo: throw
        }
    }

    std::unique_ptr<tensorrt_llm_backend_t> create_backend_from_engine_folder(const rust::Str engines_folder, const rust::Str executor_worker_path) {
        std::call_once(backend_initialized_flag, initialize_tensorrt_llm_backend);
        return std::make_unique<tensorrt_llm_backend_t>(
            std::filesystem::path(std::string_view(engines_folder.begin(), engines_folder.end()), std::filesystem::path::format::auto_format),
            std::filesystem::path(std::string_view(executor_worker_path.begin(), executor_worker_path.end()), std::filesystem::path::format::auto_format)
        );
    }
}
#endif
