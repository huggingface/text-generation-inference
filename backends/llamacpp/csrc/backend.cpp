//
// Created by Morgan Funtowicz on 9/28/2024.
//

#include <filesystem>
#include <span>

#include <ggml.h>
#include <llama.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "backend.hpp"

namespace huggingface::tgi::backends::llamacpp {

    void llama_batch_fill_prompt(llama_batch &batch, std::span<const llama_token> input_tokens) {
        for (auto i = 0; i < input_tokens.size(); ++i) {
            batch.token[i] = input_tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = nullptr;
            batch.logits[i] = false;
            ++batch.n_tokens;
        }

        batch.logits[batch.n_tokens] = true;
    }

    std::unique_ptr<llama_sampler> sampling_params_t::into_llama_sampler(const llama_model *model) const {
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
        llama_sampler_chain_add(pSampler, llama_sampler_init_top_k(static_cast<int32_t>(top_k)));

        if (0 < top_p && top_p < 1) {
            llama_sampler_chain_add(pSampler, llama_sampler_init_top_p(top_p, 1));
        }

        llama_sampler_chain_add(pSampler, llama_sampler_init_dist(seed));
        return std::unique_ptr<llama_sampler>(pSampler);
    }

    worker_t::worker_t(std::shared_ptr<llama_model> model, const llama_context_params &params)
            : mModel_(model), mParams_(params) {

#ifdef TGI_LLAMACPP_BACKEND_DEBUG
        char modelName[256];
        llama_model_meta_val_str(model.get(), "general.name", modelName, sizeof(modelName));
        SPDLOG_DEBUG(FMT_STRING("Created llama.cpp backend for model: '{}'"), std::string_view(modelName));
#endif
    }

    void worker_t::loop(std::stop_source &driver, std::queue<generation_context_t> &backlog) const {
        auto *context = llama_new_context_with_model(mModel_.get(), mParams_);

        while (!driver.stop_requested()) {
            const auto generation_context = backlog.front();

            generate(context, generation_context, std::nullopt);
            backlog.pop();

            SPDLOG_DEBUG("Processed request ({:d} remaining)", backlog.size());
        }

        llama_free(context);
    }

    size_t worker_t::generate(
            llama_context *context,
            const generation_context_t &generation_context,
            const std::optional<llama_decode_callback> &callback) const {
        // Store information about context and generation size
        auto max_new_tokens = generation_context.generation_params.max_new_tokens;

        // Convert sampling params to what llama.cpp is looking for
        auto sampler = generation_context.sampling_params.into_llama_sampler(mModel_.get());

        // Set up the prompt
        auto copy = std::vector(generation_context.input_tokens.begin(), generation_context.input_tokens.end());
        auto batch = llama_batch_get_one(copy.data(), copy.size());

        // Decode
        auto n_decoded_tokens = 0;
        for (bool generating = true; generating && n_decoded_tokens < max_new_tokens; ++n_decoded_tokens) {
            const auto callback_ = callback.value_or(llama_void_callback);

#ifdef TGI_LLAMACPP_BACKEND_DEBUG
            const auto start = std::chrono::steady_clock::now();
            const auto status = llama_decode(context, batch);
            const auto end = std::chrono::steady_clock::now();
            const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            SPDLOG_DEBUG(FMT_STRING("Successfully decoded {:d} token(s) in {}"), batch.n_tokens, latency);
#else
            const auto status = llama_decode(ctx, batch);
#endif
            batch.n_tokens = 0;
            if (LLAMA_SUCCESS(status)) {
                // Sample the new token
                auto new_token_id = llama_sampler_sample(sampler.get(), context, -1);
                auto is_eos = llama_token_is_eog(mModel_.get(), new_token_id);

                generation_context.generated_tokens[n_decoded_tokens] = new_token_id;
                generating = !is_eos;

                // Bubble up the generated token if a callback is provided
                std::invoke(std::forward<const llama_decode_callback>(callback_), new_token_id, is_eos);

                batch = llama_batch_get_one(&new_token_id, 1);
            }
        }

        return n_decoded_tokens;
    }


    backend_base_t::backend_base_t(llama_model *model) : mModel_(model, llama_free_model) { llama_backend_init(); }

    backend_base_t::~backend_base_t() { llama_backend_free(); }

    std::expected<std::vector<llama_token>, backend_error_t> backend_base_t::generate(
            std::span<const llama_token> tokens,
            const generation_params_t &generation_params,
            const sampling_params_t &sampling_params,
            const std::optional<llama_decode_callback> &callback
    ) {
        // TODO: Should we provide a way to change this value?
        auto generated = std::vector<llama_token>(2 << 8);

        auto nTokensGenerated = generate(tokens, generated, generation_params, sampling_params, callback);
        if (nTokensGenerated.has_value())
            generated.resize(*nTokensGenerated);
        return generated;
    }


    /** Single worker_t Backend impl **/

    single_worker_backend_t::single_worker_backend_t(llama_model *model,
                                                     const std::optional<llama_context_params> &params)
            : backend_base_t(model),
              mContext_(llama_context_factory(model)),
              mWorker_(mModel_, params.value_or(llama_context_default_params())) {
        llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_NUMACTL);
    }

    std::expected<std::size_t, backend_error_t>
    single_worker_backend_t::generate(
            std::span<const llama_token> tokens,
            std::span<llama_token> out,
            const generation_params_t &generation_params,
            const sampling_params_t &sampling_params,
            const std::optional<llama_decode_callback> &callback
    ) {
        return mWorker_.generate(mContext_.get(), {generation_params, sampling_params, tokens, out}, callback);
    }

    std::expected<size_t, backend_error_t>
    multi_worker_backend_t::generate(
            std::span<const llama_token>,
            std::span<llama_token>,
            const generation_params_t &generation_params,
            const sampling_params_t &sampling_params,
            const std::optional<llama_decode_callback> &callback) {
        SPDLOG_ERROR("Not implemented yet");
        return 0uz;
    }
}