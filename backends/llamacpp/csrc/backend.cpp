//
// Created by Morgan Funtowicz on 9/28/2024.
//

#include <expected>
#include <filesystem>
#include <ggml.h>
#include <llama.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>
#include "backend.hpp"

namespace huggingface::tgi::backends::llama {

    std::expected<std::unique_ptr<TgiLlamaCppBackend>, TgiLlamaCppBackendError>
    CreateLlamaCppBackend(const std::filesystem::path& modelPath) {
        SPDLOG_INFO(FMT_STRING("Loading model from {}"), modelPath);
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
            .attention_type = llama_attention_type::LLAMA_ATTENTION_TYPE_CAUSAL,
            .flash_attn = true,
        });

        return std::make_unique<huggingface::tgi::backends::llama::TgiLlamaCppBackend>(model, context);
    }

    huggingface::tgi::backends::llama::TgiLlamaCppBackend::TgiLlamaCppBackend(llama_model *const model, llama_context *const ctx)
            : model(model), ctx(ctx), batch() {
        char modelName[128];
        llama_model_meta_val_str(model, "general.name", modelName, sizeof(modelName));
        SPDLOG_DEBUG(FMT_STRING("Created llama.cpp backend for model: '{}'"), std::string_view(modelName));
    }

    huggingface::tgi::backends::llama::TgiLlamaCppBackend::~TgiLlamaCppBackend() {
        if (model) {
            SPDLOG_DEBUG("Freeing llama.cpp model");
            llama_free_model(model);
        }

        if (ctx) {
            SPDLOG_DEBUG("Freeing llama.cpp context");
            llama_free(ctx);
        }
    }

    void huggingface::tgi::backends::llama::TgiLlamaCppBackend::schedule() {
        std::vector<llama_token> tokens;
    }

    namespace impl {
        class LlamaCppBackendImpl {

        };
    }
}