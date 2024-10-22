//
// Created by Morgan Funtowicz on 9/28/2024.
//
#ifndef TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
#define TGI_LLAMA_CPP_BACKEND_BACKEND_HPP

#include <filesystem>
#include <memory>
#include <llama.h>

namespace huggingface::tgi::backends::llama {
//    const char* TGI_BACKEND_LLAMA_CPP_NAME = "llama.cpp";

    enum TgiLlamaCppBackendError {
        MODEL_FILE_DOESNT_EXIST = 1
    };


    class TgiLlamaCppBackend {
    private:
        llama_model* model;
        llama_context* ctx;
        llama_batch batch;
    public:
        TgiLlamaCppBackend(llama_model *model, llama_context *ctx);
        ~TgiLlamaCppBackend();

        void schedule();
    };

    std::expected<std::unique_ptr<TgiLlamaCppBackend>, TgiLlamaCppBackendError>
    CreateLlamaCppBackend(const std::filesystem::path& root);
}

#endif //TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
