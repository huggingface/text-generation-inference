//
// Created by Morgan Funtowicz on 9/28/2024.
//
#ifndef TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
#define TGI_LLAMA_CPP_BACKEND_BACKEND_HPP

#include <cmath>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <llama.h>

#define LLAMA_SUCCESS(x) x == 0

namespace huggingface::tgi::backends::llama {
    enum TgiLlamaCppBackendError: uint8_t {
        MODEL_FILE_DOESNT_EXIST = 1
    };

    class TgiLlamaCppBackend {
        using TokenId = llama_token;

    private:
        llama_model* model;
        llama_context* ctx;

        /**
         *
         * @param topK
         * @param topP
         * @return
         */
        std::unique_ptr<llama_sampler *> GetSamplerFromArgs(
                uint32_t topK, float_t topP, float_t frequencyPenalty, float_t repetitionPenalty, uint64_t seed);

    public:
        TgiLlamaCppBackend(llama_model *model, llama_context *ctx);
        ~TgiLlamaCppBackend();

        /**
         *
         * @param text
         * @return
         */
        [[nodiscard("Tokens will be freed after this call if not assigned to an lvalue")]]
        std::vector<TgiLlamaCppBackend::TokenId> Tokenize(const std::string& text) const;

        /**
         *
         * @param tokens
         * @param topK
         * @param topP
         * @param frequencyPenalty
         * @param repetitionPenalty
         * @param maxNewTokens
         * @param seed
         * @return
         */
        [[nodiscard("Generated tokens will be freed after this call if not assigned to an lvalue")]]
        std::expected<std::vector<TgiLlamaCppBackend::TokenId>, TgiLlamaCppBackendError> Generate(
                std::span<const TokenId> tokens,
                uint32_t topK,
                float_t topP = 1.0f,
                float_t frequencyPenalty = 0.0f,
                float_t repetitionPenalty = 0.0f,
                uint32_t maxNewTokens = std::numeric_limits<uint32_t>::max() - 1,
                uint64_t seed = 2014
        );
    };

    [[nodiscard("Create backend will be freed after this call if not assigned to an lvalue")]]
    std::expected<std::unique_ptr<TgiLlamaCppBackend>, TgiLlamaCppBackendError>
    CreateLlamaCppBackend(const std::filesystem::path& root);
}

#endif //TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
