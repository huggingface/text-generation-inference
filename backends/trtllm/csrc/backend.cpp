#include <ranges>
#include <utility>
#include "backend.hpp"

#include <spdlog/spdlog.h>

namespace huggingface::tgi::backends::trtllm {

    size_t backend_t::num_tokens_ready() const noexcept {
        return executor_.getNumResponsesReady();
    }

    std::expected<request_id_t, backend_exception_t>
    backend_t::submit(std::span<tle::TokenIdType> token_ids, generation_params_t generation_params, sampling_params_t sampling_params) noexcept {
        SPDLOG_DEBUG(FMT_STRING("Submitting {:d} tokens to the executor for scheduling"), token_ids.size());
        return executor_.enqueueRequest(tle::Request {
                {token_ids.begin(), token_ids.end()},  // Making actual copy of the tokens
                static_cast<tle::SizeType32>(generation_params.max_new_tokens),
                true,
                (tle::SamplingConfig) sampling_params,
                tle::OutputConfig { /* returnLogProbs= */ true },
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                stop_words_
        });
    }

    std::vector<tle::Response> backend_t::pull_tokens() noexcept {
        return executor_.awaitResponses();
    }

    void backend_t::cancel(request_id_t request_id) noexcept {
        SPDLOG_INFO(FMT_STRING("Cancelling request: {:d}"), request_id);
        executor_.cancelRequest(request_id);
    }
}