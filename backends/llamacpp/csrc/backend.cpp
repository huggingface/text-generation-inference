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

    std::unique_ptr<llama_sampler> SamplingParams::IntoLlamaSampler(const llama_model *pModel) const {
        auto *pSampler = llama_sampler_chain_init({.no_perf = false});

        // Penalties
        llama_sampler_chain_add(pSampler, llama_sampler_init_penalties(
                llama_n_vocab(pModel),
                llama_token_eos(pModel),
                llama_token_nl(pModel),
                0.0f,
                repetitionPenalty,
                frequencyPenalty,
                0.0f,
                false,
                false
        ));
        llama_sampler_chain_add(pSampler, llama_sampler_init_top_k(static_cast<int32_t>(topK)));

        if (0 < topP && topP < 1) {
            llama_sampler_chain_add(pSampler, llama_sampler_init_top_p(topP, 1));
        }

        llama_sampler_chain_add(pSampler, llama_sampler_init_dist(seed));
        return std::unique_ptr<llama_sampler>(pSampler);
    }

    Worker::Worker(std::shared_ptr<llama_model> pModel, const llama_context_params &params)
            : mModel_(pModel), mParams_(params) {

#ifdef TGI_LLAMACPP_BACKEND_DEBUG
        char modelName[256];
        llama_model_meta_val_str(pModel.get(), "general.name", modelName, sizeof(modelName));
        SPDLOG_DEBUG(FMT_STRING("Created llama.cpp backend for model: '{}'"), std::string_view(modelName));
#endif
    }

    void Worker::Loop(std::atomic_flag &running, std::atomic_uint8_t &waiting, std::queue<SamplingParams> &backlog) {
        auto *context = llama_new_context_with_model(mModel_.get(), mParams_);

        while (running.test(std::memory_order_acquire)) {
            if (waiting.load(std::memory_order_acquire) > 0) {
                --waiting;

                auto request = backlog.front();
                auto sampler = request.IntoLlamaSampler(mModel_.get());

                // Retrieve decoding context
                auto batch = llama_batch_get_one(tokens.data(), tokens.size());
                // Decode
                for (auto [generating, nDecoded] = std::pair{true, 0uz}; generating && nDecoded < 1; ++nDecoded) {
#ifdef TGI_LLAMACPP_BACKEND_DEBUG
                    const auto start = std::chrono::steady_clock::now();
                    const auto status = llama_decode(context, batch);
                    const auto end = std::chrono::steady_clock::now();
                    const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    SPDLOG_DEBUG(FMT_STRING("Successfully decoded {:d} token(s) in {}"), batch.n_tokens, latency);
#else
                    const auto status = llama_decode(ctx, batch);
#endif
                    if (LLAMA_SUCCESS(status)) {
                        // Sample the new token
                        auto new_token_id = llama_sampler_sample(sampler.get(), context, -1);
                        generated.emplace_back(new_token_id);
                        generating = !llama_token_is_eog(mModel_.get(), new_token_id);

                        // Next iteration
                        batch = llama_batch_get_one(&new_token_id, 1);
                    }
                }

                backlog.pop();

            }
        }

        llama_free(context);
    }

    huggingface::tgi::backends::llamacpp::BackendBase::BackendBase(llama_model *model)
            : mModel_(model, llama_free_model) { llama_backend_init(); }

    BackendBase::~BackendBase() { llama_backend_free(); }
}