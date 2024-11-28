//
// Created by Morgan Funtowicz on 9/28/2024.
//

#include <filesystem>

#include <ggml.h>
#include <llama.h>
#include <spdlog/fmt/chrono.h>
#include <spdlog/spdlog.h>

#include "backend.hpp"

namespace huggingface::tgi::backends::llamacpp {

    llama_sampler_ptr sampling_params_t::into_llama_sampler(const llama_model *model) const {
        auto *pSampler = llama_sampler_chain_init({.no_perf = false});

        // Penalties
        llama_sampler_chain_add(pSampler, llama_sampler_init_penalties(
                llama_n_vocab(model),
                llama_token_eos(model),
                llama_token_nl(model),
                0.0f,
                repetition_penalty,
                frequency_penalty,
                0.0f,
                false,
                false
        ));

        if (top_k > 0) {
            llama_sampler_chain_add(pSampler, llama_sampler_init_top_k(static_cast<int32_t>(top_k)));
        }

        if (0 < top_p && top_p < 1) {
            llama_sampler_chain_add(pSampler, llama_sampler_init_top_p(top_p, 1));
        }

        llama_sampler_chain_add(pSampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(pSampler, llama_sampler_init_dist(seed));
        return {pSampler, llama_sampler_deleter};
    }


    std::expected<llama_batch, backend_error_t> get_batch_from_prompt(std::span<llama_token> prompt) {
        auto batch = llama_batch_init(static_cast<int32_t>(prompt.size()), 0, 1);
        std::for_each(prompt.begin(), prompt.end(), [&batch](const llama_token token) {
            const auto n_token = batch.n_tokens;

            batch.token[n_token] = token;
            batch.pos[n_token] = n_token;
            batch.n_seq_id[n_token] = 1;
            batch.seq_id[n_token][0] = 1;
            batch.logits[n_token] = false;
            batch.n_tokens++;
        });

        batch.logits[batch.n_tokens - 1] = true;
        return batch;
    }

    void update_batch_for_decoding(llama_batch &batch, llama_token token, size_t position) {
        batch.n_tokens = 1;
        batch.logits[0] = true;
        batch.token[0] = token;
        batch.pos[0] = static_cast<int32_t>(position);
    }

    worker_t::worker_t(std::shared_ptr<llama_model> model, const llama_context_params &&params)
            : model_(model), context_(llama_new_context_with_model(model_.get(), params)) {

#ifdef TGI_LLAMACPP_BACKEND_DEBUG
        char modelName[256];
        llama_model_meta_val_str(model.get(), "general.name", modelName, sizeof(modelName));
        SPDLOG_DEBUG(FMT_STRING("Created llama.cpp backend for model: '{}'"), std::string_view(modelName));
#endif
    }

    std::expected<size_t, backend_error_t>
    worker_t::generate(const generation_context_t &generation_context,
                       const std::optional<llama_decode_callback> &callback) const {
        // Store information about context and generation size
        const auto callback_ = callback.value_or(llama_void_callback);
        auto max_new_tokens = generation_context.generation_params.max_new_tokens;

        // Convert sampling params to what llama.cpp is looking for
        auto sampler = generation_context.sampling_params.into_llama_sampler(model_.get());

        // Set up the prompt
        if (auto maybe_batch = get_batch_from_prompt(generation_context.input_tokens); maybe_batch.has_value()) {
            // Decode
            auto batch = *maybe_batch;
            auto n_decoded_tokens = 0;
            const auto prompt_size = generation_context.input_tokens.size();
            for (bool generating = true; generating; ++n_decoded_tokens) {

#ifdef TGI_LLAMACPP_BACKEND_DEBUG
                const auto start = std::chrono::steady_clock::now();
                const auto status = llama_decode(context_.get(), batch);
                const auto end = std::chrono::steady_clock::now();
                const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                SPDLOG_DEBUG(FMT_STRING("Successfully decoded {:d} token(s) in {}"), batch.n_tokens, latency);
#else
                const auto status = llama_decode(context_.get(), batch);
#endif
                batch.n_tokens = 0;
                if (LLAMA_SUCCESS(status)) [[likely]] {
                    // Sample the new token
                    auto new_token_id = llama_sampler_sample(sampler.get(), context_.get(), -1);
                    const auto is_eog = llama_token_is_eog(model_.get(), new_token_id);
                    const auto new_token_logits = llama_get_logits_ith(context_.get(), -1); // TODO: return logit

                    // Handle termination cases
                    const bool has_reach_max_tokens = n_decoded_tokens >= max_new_tokens - 1;
                    const bool has_reach_eog = !generation_context.generation_params.ignore_eos_token & is_eog;
                    const bool is_final = has_reach_max_tokens | has_reach_eog;

                    // Bubble up the generated token if a callback is provided
                    const auto should_stop = callback_(new_token_id, *new_token_logits, is_final, n_decoded_tokens + 1);

                    // Compute the continuation flag
                    generating = !(should_stop | is_final);

                    // Update the batch for the next generation
                    update_batch_for_decoding(batch, new_token_id, prompt_size + n_decoded_tokens);
                }
            }

            llama_batch_free(batch);

            return n_decoded_tokens;
        } else {
            return maybe_batch.error();
        }
    }
}