//
// Created by mfuntowicz on 10/23/24.
//

#ifndef TGI_LLAMA_CPP_BACKEND_FFI_HPP
#define TGI_LLAMA_CPP_BACKEND_FFI_HPP

#include <exception>
#include <filesystem>
#include <memory>
#include <string_view>
#include <variant>

#include <spdlog/spdlog.h>

namespace huggingface::tgi::backends::llamacpp {
    class llama_cpp_worker_frontend_t;
}

#include "backend.hpp"
#include "backends/llamacpp/src/lib.rs.h"
#include "rust/cxx.h"


namespace huggingface::tgi::backends::llamacpp {

    auto llama_model_deleter = [](llama_model *model) { llama_free_model(model); };
    auto make_shared_llama_model = [](llama_model *model) {
        return std::shared_ptr<llama_model>(model, llama_model_deleter);
    };

    class llama_cpp_backend_exception_t : std::exception {};

    /**
     * Llama.cpp frontend over the worker interfacing with Rust FFI layer
     */
    class llama_cpp_worker_frontend_t {
    private:
        std::shared_ptr<llama_model> model_;
        worker_t worker_;

    public:
        explicit llama_cpp_worker_frontend_t(llama_model *model):
            model_{ make_shared_llama_model(model) }, worker_(model_, {.no_perf = true}) {}

        size_t stream(
                rust::Slice<const uint32_t> input_tokens,
                const generation_params_t generation_params,
                const sampling_params_t &sampling_params,
                InferContext *ctx,
                rust::Fn<bool(InferContext *, uint32_t, float_t, bool, size_t)> callback
        ) {
            auto context_forwarding_callback =
                    [=, &ctx](uint32_t new_token_id, float_t logits, bool is_eos, size_t n_generated_tokens) -> bool {
                return callback(ctx, new_token_id, logits, is_eos, n_generated_tokens);
            };

            // Ask the compiler to create view over Rust slice transmuting from uint32_t* to llama_token*
            auto input_tokens_v = std::vector<llama_token>(input_tokens.size());
            std::memcpy(input_tokens_v.data(), input_tokens.data(), input_tokens.size());

            const auto generation_context = generation_context_t {generation_params, sampling_params, input_tokens_v};
            if(const auto result = worker_.generate(generation_context, context_forwarding_callback); result.has_value()) [[likely]] {
                return *result;
            } else {
                throw llama_cpp_backend_exception_t {};
            }
        }
    };

    std::unique_ptr<llama_cpp_worker_frontend_t> create_worker_frontend(rust::Str modelPath) {
        const auto cxxPath = std::string(modelPath);
        auto params = llama_model_default_params();
        params.use_mmap = true;

        auto *model = (llama_load_model_from_file(cxxPath.c_str(), params));
        return std::make_unique<llama_cpp_worker_frontend_t>(model);
    }
}


#endif //TGI_LLAMA_CPP_BACKEND_FFI_HPP
