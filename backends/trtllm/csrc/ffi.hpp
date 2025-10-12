#ifndef TGI_BACKEND_TRTLLM_FFI
#define TGI_BACKEND_TRTLLM_FFI

#include <chrono>
#include <exception>
#include <memory>
#include <optional>
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
    } catch (const std::exception &e) {
        fail(e.what());
    }
}

namespace huggingface::tgi::backends::trtllm {
    class tensorrt_llm_backend_t;
}

#include "backends/trtllm/src/lib.rs.h"


namespace huggingface::tgi::backends::trtllm {
    std::once_flag backend_initialized_flag;

    constexpr finish_reason_t as_finish_reason_t(const tle::FinishReason reason) noexcept {
        switch (reason) {
            case tle::FinishReason::kNOT_FINISHED:
                return finish_reason_t::kNOT_FINISHED;
            case tle::FinishReason::kSTOP_WORDS:
                return finish_reason_t::kSTOP_WORDS;
            case tle::FinishReason::kEND_ID:
                return finish_reason_t::kEND_ID;
            case tle::FinishReason::kLENGTH:
                return finish_reason_t::kLENGTH;
            case tle::FinishReason::kTIMED_OUT:
                return finish_reason_t::kTIMED_OUT;
            case tle::FinishReason::kCANCELLED:
                return finish_reason_t::kCANCELLED;
            default:
                std::unreachable();
        }
    }

    static auto as_generation_step = [](const tle::Response &r, const std::chrono::time_point<std::chrono::steady_clock> created) {
        const auto reqId = r.getRequestId();
        if (!r.hasError()) [[likely]] {
            const auto result = r.getResult();
            std::optional<uint32_t> token_id = std::nullopt;
            if (!result.outputTokenIds.empty() && !result.outputTokenIds[0].empty()) {
                token_id = static_cast<uint32_t>(result.outputTokenIds[0][0]);
            }

            std::optional<float> log_prob = std::nullopt;
            if (result.logProbs && !result.logProbs->empty() && !result.logProbs.value()[0].empty()) {
                log_prob = result.logProbs.value()[0].back();
            }

            std::optional<int64_t> first_scheduled_time_ns = std::nullopt;
            if (result.requestPerfMetrics) {
                const auto &t = result.requestPerfMetrics->timingMetrics;
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t.firstScheduledTime - created).count();
                first_scheduled_time_ns = static_cast<int64_t>(ns);
            }

            return generation_step_t{
                    reqId,
                    token_id.value_or(0),
                    log_prob.value_or(0.0),
                    first_scheduled_time_ns.value_or(0),
                    result.isFinal,
                    as_finish_reason_t(result.finishReasons[0]),
                    token_id.has_value(),
                    log_prob.has_value(),
                    first_scheduled_time_ns.has_value(),
                    false,
                    std::string()
            };
        } else {
            return generation_step_t{
                    reqId,
                    0,
                    0.0,
                    0,
                    true,
                    finish_reason_t::kNOT_FINISHED,
                    false,
                    false,
                    false,
                    true,
                    std::move(r.getErrorMsg())
            };
        }
    };


    class tensorrt_llm_backend_t {
    private:
        mutable backend_t inner_;

        // m_created_time is a reference point to convert time from c++ time_point
        // to rust Instant.
        std::chrono::time_point<std::chrono::steady_clock> m_created_time;


    public:
        tensorrt_llm_backend_t(std::filesystem::path &&engine_folder, std::filesystem::path &&executor_worker_path, const std::chrono::time_point<std::chrono::steady_clock>& created_time, const std::vector<std::string>& encoded_vocab, std::string_view tokenizer_str)
                : inner_(engine_folder, executor_worker_path, encoded_vocab, tokenizer_str),
                  m_created_time {created_time}
        {}

        request_id_t submit(
                rust::Slice<const uint32_t> tokens,
                uint32_t max_new_tokens,
                uint32_t top_k,
                float_t top_p,
                float_t temperature,
                float_t repetition_penalty,
                float_t frequency_penalty,
                uint64_t seed,
                grammar_type_t grammar_type,
                rust::Str grammar_value
        ) const {
            // This is enabled only if using add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
            SPDLOG_TRACE(FMT_STRING("[FFI] Submitting {:d} prompt tokens to the executor"));

            // Submit the request to the executor and get back a potential request_id used to track request status
            const auto signed_tokens = std::vector<int32_t>(tokens.begin(), tokens.end());

            std::optional<tle::GuidedDecodingParams::GuideType> guide_type = std::nullopt;
            switch (grammar_type) {
            case grammar_type_t::kJSON:
                guide_type = tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA;
                break;
            case grammar_type_t::kREGEX:
                guide_type = tle::GuidedDecodingParams::GuideType::kREGEX;
                break;
            default:
                break;
            }

            const auto maybe_request_id = inner_.submit(
                    signed_tokens,
                    {max_new_tokens, guide_type, std::string(grammar_value)},
                    {top_k, top_p, repetition_penalty, frequency_penalty, temperature, seed}
            );

            // If we do have a value, let's return the request_id
            if (maybe_request_id.has_value()) [[likely]] {
                return *maybe_request_id;
            } else {
                SPDLOG_WARN("[FFI] Failed to submit request to the executor");
                return maybe_request_id.error();
            }
        }

        std::unique_ptr<std::vector<generation_step_t>> pull_tokens() const noexcept {
            const auto responses = inner_.pull_tokens();

            SPDLOG_TRACE("[FFI] Successfully pulled out {:d} responses from executor", responses.size());

            auto f = [this](const tle::Response &r){
                return as_generation_step(r, m_created_time);
            };
            auto steps = std::make_unique<std::vector<generation_step_t>>();
            // Transform tle::Response to generation_step_t
#ifdef __cpp_lib_ranges_to_container
            *steps = responses | std::views::transform(f) | std::ranges::to<std::vector>();
#else
            steps->reserve(responses.size());
            std::transform(responses.begin(), responses.end(), std::back_inserter(steps), f);
#endif
            return steps;
        }

        void cancel(request_id_t request_id) const noexcept {
            SPDLOG_DEBUG("[FFI] cancelling request {:d}", request_id);
            inner_.cancel(request_id);
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

    std::unique_ptr<tensorrt_llm_backend_t>
    create_backend_from_engine_folder(const rust::Str engines_folder, const rust::Str executor_worker_path, const rust::Str tokenizer_str, const rust::Vec<rust::String> encoded_vocab) {
        const auto created_time = std::chrono::steady_clock::now();
        std::call_once(backend_initialized_flag, initialize_tensorrt_llm_backend);

        std::vector<std::string> encoded_vocab_std{};
        encoded_vocab_std.reserve(encoded_vocab.size());

        for (const auto& v : encoded_vocab) {
            encoded_vocab_std.push_back(std::string(v));
        }

        return std::make_unique<tensorrt_llm_backend_t>(
                std::filesystem::path(std::string_view(engines_folder.begin(), engines_folder.end()),
                                      std::filesystem::path::format::auto_format),
                std::filesystem::path(std::string_view(executor_worker_path.begin(), executor_worker_path.end()),
                                      std::filesystem::path::format::auto_format),
                created_time,
                encoded_vocab_std,
                std::string_view(tokenizer_str)
        );
    }
}
#endif
