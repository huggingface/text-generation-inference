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

namespace huggingface::tgi::backends::llamacpp::impl {
    class LlamaCppBackendImpl;
}


#include "backends/llamacpp/src/lib.rs.h"


namespace huggingface::tgi::backends::llamacpp::impl {

    class LlamaCppBackendException : std::exception {

    };

    class LlamaCppBackendImpl {
    private:
        TgiLlamaCppBackend _inner;

    public:
        LlamaCppBackendImpl(llama_model *model, llama_context *context) : _inner(model, context) {}
    };

    std::unique_ptr<LlamaCppBackendImpl> CreateLlamaCppBackendImpl(rust::Str modelPath) {
        const auto cxxPath = std::string_view(modelPath);
        if (auto maybe = TgiLlamaCppBackend::FromGGUF(std::filesystem::path(cxxPath)); maybe.has_value()) {
            auto [model, context] = *maybe;
            return std::make_unique<LlamaCppBackendImpl>(model, context);
        } else {
            throw LlamaCppBackendException();
        }
    }
}


#endif //TGI_LLAMA_CPP_BACKEND_FFI_HPP
