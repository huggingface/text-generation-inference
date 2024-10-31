//
// Created by mfuntowicz on 10/23/24.
//

#ifndef TGI_LLAMA_CPP_BACKEND_FFI_HPP
#define TGI_LLAMA_CPP_BACKEND_FFI_HPP

#include <exception>
#include <filesystem>
#include <string_view>

#include <spdlog/spdlog.h>
#include "backend.hpp"

namespace huggingface::tgi::backends::llamacpp {
    struct generation_params_t;
    struct sampling_params_t;

    class llama_cpp_backend_impl_t;
}


#include "backends/llamacpp/src/lib.rs.h"


namespace huggingface::tgi::backends::llamacpp {

    // Concept identifying types which have a .generate() -> size_t method to do in-place generation
    template<typename T>
    concept has_emplace_generate = requires(
            T t,
            std::span<const llama_token> input_tokens,
            std::span<llama_token> generated_tokens,
            const generation_params_t &generation_params,
            const sampling_params_t &sampling_params,
            llama_decode_callback callback
    ) {
        {
        t.generate(input_tokens, generated_tokens, generation_params, sampling_params, callback)
        } -> std::same_as<std::expected<size_t, backend_error_t>>;
    };

    static_assert(has_emplace_generate<single_worker_backend_t>,
                  "single_worker_backend_t doesn't meet concept is_generate_emplace_capable");
    static_assert(has_emplace_generate<multi_worker_backend_t>,
                  "multi_worker_backend_t doesn't meet concept is_generate_emplace_capable");

    class llama_cpp_backend_exception_t : std::exception {

    };

    /**
     * Llama.cpp backend interfacing with Rust FFI layer
     */
    class llama_cpp_backend_impl_t {
    private:
        std::variant<single_worker_backend_t, multi_worker_backend_t> mInner_;

    public:
        explicit llama_cpp_backend_impl_t(single_worker_backend_t &&backend) : mInner_(std::move(backend)) {}

        explicit llama_cpp_backend_impl_t(multi_worker_backend_t &&backend) : mInner_(std::move(backend)) {}

        size_t generate(
                rust::Slice<const uint32_t> input_tokens,
                rust::Slice <uint32_t> generated_tokens,
                const generation_params_t &generation_params,
                const sampling_params_t &sampling_params,
                rust::Fn<void(uint32_t, bool)> callback
        ) {
            // Define the visitor lambda function which requires the has_emplace_generate constraint on T
            static auto inner_fw = [=, &generation_params, &sampling_params]<has_emplace_generate T>(T &&backend)
                    -> std::expected<size_t, backend_error_t> {

                // Ask the compiler to create view over Rust slice transmuting from uint32_t* to int32_t*
                auto input_tokens_v =
                        std::span(reinterpret_cast<const llama_token *>(input_tokens.data()), input_tokens.size());
                auto generated_tokens_v =
                        std::span(reinterpret_cast<llama_token *>(generated_tokens.data()), generated_tokens.size());

                return backend.generate(
                        input_tokens_v, generated_tokens_v, generation_params, sampling_params, callback);
            };

            if (const auto result = std::visit(inner_fw, mInner_); result.has_value()) {
                return *result;
            } else {
                throw llama_cpp_backend_exception_t();
            }
        }
    };

    std::unique_ptr<llama_cpp_backend_impl_t> create_single_worker_backend(rust::Str modelPath) {
        const auto cxxPath = std::string(modelPath);
        auto params = llama_model_default_params();
        params.use_mmap = true;

        auto *model = llama_load_model_from_file(cxxPath.c_str(), params);
        auto backend = single_worker_backend_t(model, std::nullopt);
        return std::make_unique<llama_cpp_backend_impl_t>(std::move(backend));
    }
}


#endif //TGI_LLAMA_CPP_BACKEND_FFI_HPP
