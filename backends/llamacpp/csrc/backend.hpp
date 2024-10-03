//
// Created by Morgan Funtowicz on 9/28/2024.
//

#ifndef TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
#define TGI_LLAMA_CPP_BACKEND_BACKEND_HPP

#include <memory>
#include <llama.h>

namespace huggingface::tgi::backends::llama {
    const char* TGI_BACKEND_LLAMA_CPP_NAME = "llama.cpp";


    class TgiLlamaCppBackend {
    private:
        llama_model* model;
        llama_context* ctx;
        llama_batch batch;
    public:
        TgiLlamaCppBackend(llama_model* const model, llama_context* const);
        ~TgiLlamaCppBackend();
    };

    std::unique_ptr<TgiLlamaCppBackend> CreateLlamaCppBackend(std::string_view root);
}

#endif //TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
