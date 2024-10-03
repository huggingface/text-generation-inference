//
// Created by Morgan Funtowicz on 9/28/2024.
//

#include <arg.h>
#include <common.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "backend.hpp"

namespace huggingface::tgi::backends::llama {

    std::unique_ptr<TgiLlamaCppBackend> CreateLlamaCppBackend(std::string_view root) {
        SPDLOG_INFO(FMT_STRING("Loading model from {}"), root);
        gpt_init();

        // Fake argv
        std::vector<std::string_view> args = {"tgi_llama_cpp_backend", "--model", root};
        std::vector<char*> argv;
        for(const auto& arg : args) {
            argv.push_back(const_cast<char *>(arg.data()));
        }
        argv.push_back(nullptr);

        // Create the GPT parameters
        gpt_params params;
        if (!gpt_params_parse(args.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER)) {
            throw std::runtime_error("Failed to create GPT Params from model");
        }


        // Create the inference engine
        SPDLOG_INFO("Allocating llama.cpp model from gpt_params");
        auto result = llama_init_from_gpt_params(params);

        // Unpack all the inference engine components
        auto model = result.model;
        auto context = result.context;
        auto loras = result.lora_adapters;

        // Make sure everything is correctly initialized
        if(model == nullptr)
            throw std::runtime_error(fmt::format("Failed to load model from {}", root));

        return std::make_unique<TgiLlamaCppBackend>(model, context);
    }

    TgiLlamaCppBackend::TgiLlamaCppBackend(llama_model *const model, llama_context *const ctx)
        : model(model), ctx(ctx), batch() {

    }

    TgiLlamaCppBackend::~TgiLlamaCppBackend() {
        if(model)
        {
            SPDLOG_DEBUG("Freeing llama.cpp model");
            llama_free_model(model);
        }

        if(ctx)
        {
            SPDLOG_DEBUG("Freeing llama.cpp context");
            llama_free(ctx);
        }
    }
}