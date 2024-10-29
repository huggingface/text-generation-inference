//
// Created by Morgan Funtowicz on 9/28/2024.
//
#ifndef TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
#define TGI_LLAMA_CPP_BACKEND_BACKEND_HPP

#include <atomic>
#include <cmath>
#include <expected>
#include <filesystem>
#include <queue>
#include <memory>
#include <span>
#include <vector>

#include <llama.h>

#define LLAMA_SUCCESS(x) x == 0

namespace huggingface::tgi::backends::llamacpp {
    enum BackendError : uint8_t {
        MODEL_FILE_DOESNT_EXIST = 1
    };

    struct SamplingParams {
        uint32_t topK = std::numeric_limits<decltype(topK)>::max();
        float_t topP = 1.0f;
        float_t frequencyPenalty = 0.0f;
        float_t repetitionPenalty = 0.0f;
        uint64_t seed = 2014;

        /**
         * Convert this GenerationParams to the respective llama_sampler structure
         * @param Pointer to the model data
         * @return
         */
        std::unique_ptr<llama_sampler> IntoLlamaSampler(const llama_model *) const;
    };

    class Worker {
    protected:
        constexpr static auto llama_context_deleter = [](llama_context *pContext) { llama_free(pContext); };

    public:
        using model_ptr_type = std::shared_ptr<llama_model>;
        using context_params_type = llama_context_params;
        using token_id_type = llama_token;

    private:
        const model_ptr_type mModel_;
        context_params_type mParams_;

    public:
        Worker(std::shared_ptr<llama_model> pModel, const llama_context_params &params);

        void Loop(std::atomic_flag &, std::atomic_uint8_t &, std::queue<SamplingParams> &) const;
    };


    class BackendBase {

    private:
        std::shared_ptr<llama_model> mModel_;

    public:
        explicit BackendBase(llama_model *model);

        ~BackendBase();

        /**
         *
         * @param tokens
         * @params out
         * @param params
         * @param maxNewTokens
         * @return
         */
        [[nodiscard("Generated tokens will be freed after this call if not assigned to an lvalue")]]
        std::expected<std::vector<llama_token>, BackendError> Generate(
                std::span<const llama_token> tokens,
                std::span<llama_token> out,
                const SamplingParams &params,
                uint32_t maxNewTokens = std::numeric_limits<uint32_t>::max() - 1
        );

        /**
         *
         * @param tokens
         * @param params
         * @param maxNewTokens
         * @return
         */
        [[nodiscard("Generated tokens will be freed after this call if not assigned to an lvalue")]]
        std::expected<std::vector<llama_token>, BackendError> Generate(
                std::span<const llama_token> tokens,
                const SamplingParams &params,
                uint32_t maxNewTokens = std::numeric_limits<uint32_t>::max() - 1
        );
    };
}

#endif //TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
