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

namespace huggingface::tgi::backends::llama {

    std::expected<std::unique_ptr<TgiLlamaCppBackend>, TgiLlamaCppBackendError>
    CreateLlamaCppBackend(const std::filesystem::path& modelPath) {
        SPDLOG_DEBUG(FMT_STRING("Loading model from {}"), modelPath);
        llama_backend_init();
        llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_NUMACTL);

        // Load the model
        if(!exists(modelPath)) {
            return std::unexpected(TgiLlamaCppBackendError::MODEL_FILE_DOESNT_EXIST);
        }

        auto params = llama_model_default_params();
        auto* model = llama_load_model_from_file(modelPath.c_str(), params);
        auto* context = llama_new_context_with_model(model, {
            .n_batch = 1,
            .n_threads = 16,
            .attention_type = llama_attention_type::LLAMA_ATTENTION_TYPE_CAUSAL,
            .flash_attn = false,
        });

        return std::make_unique<huggingface::tgi::backends::llama::TgiLlamaCppBackend>(model, context);
    }

    huggingface::tgi::backends::llama::TgiLlamaCppBackend::TgiLlamaCppBackend(llama_model *const model, llama_context *const ctx)
        : model(model), ctx(ctx) {
#ifndef NDEBUG
        char modelName[256];
        llama_model_meta_val_str(llama_get_model(ctx), "general.name", modelName, sizeof(modelName));
        SPDLOG_DEBUG(FMT_STRING("Created llama.cpp backend for model: '{}'"), std::string_view(modelName));
#endif
    }

    huggingface::tgi::backends::llama::TgiLlamaCppBackend::~TgiLlamaCppBackend() {
        if (ctx) {
            SPDLOG_DEBUG("Freeing llama.cpp context");
            llama_free(ctx);
        }

        if(model) {
            SPDLOG_DEBUG("Freeing llama.cpp model");
            llama_free_model(model);
        }
    }

    std::vector<TgiLlamaCppBackend::TokenId> TgiLlamaCppBackend::Tokenize(const std::string &text) const {
        std::vector<TgiLlamaCppBackend::TokenId> tokens(llama_n_seq_max(ctx));

        if(auto nTokens = llama_tokenize(model, text.c_str(), text.length(), tokens.data(), tokens.capacity(), true, true); nTokens < 0){
            tokens.resize(-nTokens);
            llama_tokenize(model, text.c_str(), text.length(), tokens.data(), tokens.capacity(), true, true);
        } else {
            tokens.resize(nTokens);
        }

        SPDLOG_DEBUG(FMT_STRING("Tokenized input with {:d} tokens"), tokens.size());
        return tokens;
    }

    std::unique_ptr<llama_sampler *> TgiLlamaCppBackend::GetSamplerFromArgs(
            const uint32_t topK, const float_t topP, const float_t frequencyPenalty, const float_t repetitionPenalty, const uint64_t seed) {
        auto *sampler = llama_sampler_chain_init({.no_perf = false});

        // Penalties
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
                llama_n_vocab(model),
                llama_token_eos(model),
                llama_token_nl (model),
                0.0f,
                repetitionPenalty,
                frequencyPenalty,
                0.0f,
                false,
                false
        ));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(static_cast<int32_t>(topK)));

        if(0 < topP && topP < 1) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
        }

        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
        return std::make_unique<llama_sampler*>(sampler);
    }

    std::vector<TgiLlamaCppBackend::TokenId> huggingface::tgi::backends::llama::TgiLlamaCppBackend::Generate(
            std::span<const TokenId> tokens, const uint32_t topK, const float_t topP, const uint32_t maxNewTokens) {
        SPDLOG_DEBUG(FMT_STRING("Received {:d} tokens to schedule"), tokens.size());

        // Allocate generation result
        std::vector<TgiLlamaCppBackend::TokenId> generated;
        generated.reserve(llama_n_seq_max(ctx) - tokens.size());

        // Retrieve decoding context
        auto batch = llama_batch_get_one(const_cast<int32_t *>(tokens.data()), static_cast<int32_t>(tokens.size()));
        auto sampler = GetSamplerFromArgs(topK, topP, 1.0, 1.0, 2014);

        // Decode
        for(auto [generating, nDecoded] = std::pair{true, 0uz}; generating && nDecoded < maxNewTokens; ++nDecoded) {
#ifndef NDEBUG
            const auto start = std::chrono::steady_clock::now();
            const auto status = llama_decode(ctx, batch);
            const auto end = std::chrono::steady_clock::now();
            const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            SPDLOG_DEBUG(FMT_STRING("Successfully decoded {:d} token(s) in {}"), batch.n_tokens, latency);
#else
            const auto status = llama_decode(ctx, batch);
#endif
            if (LLAMA_SUCCESS(status)) {
                // Sample the new token
                auto new_token_id = llama_sampler_sample(*sampler, ctx, -1);
                generated.emplace_back(new_token_id);
                generating = !llama_token_is_eog(model, new_token_id);

                // Next iteration
                batch = llama_batch_get_one(&new_token_id, 1);
            }
        }
        return generated;
    }
}