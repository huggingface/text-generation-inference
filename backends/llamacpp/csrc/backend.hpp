//
// Created by Morgan Funtowicz on 9/28/2024.
//
#ifndef TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
#define TGI_LLAMA_CPP_BACKEND_BACKEND_HPP

#include <atomic>
#include <cmath>
#include <expected>
#include <filesystem>
#include <functional>
#include <queue>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <vector>

#include <llama.h>
#include <thread>

#define LLAMA_SUCCESS(x) x == 0

namespace huggingface::tgi::backends::llamacpp {

    static constexpr auto llama_context_deleter = [](llama_context *pContext) { llama_free(pContext); };
    typedef std::unique_ptr<llama_context, decltype(llama_context_deleter)> llama_context_smart_ptr;

    typedef std::function<void(llama_token, bool)> llama_decode_callback;
    static constexpr auto llama_void_callback = [](llama_token token_id, bool is_eos) {};

    /**
     *
     */
    enum backend_error_t : uint8_t {
        MODEL_FILE_DOESNT_EXIST = 1
    };

    /**
     *
     */
    struct sampling_params_t {
        uint32_t top_k = std::numeric_limits<decltype(top_k)>::max();
        float_t top_p = 1.0f;
        float_t frequency_penalty = 0.0f;
        float_t repetition_penalty = 0.0f;
        uint64_t seed = 2014;

        /**
         * Convert this GenerationParams to the respective llama_sampler structure
         * @param Pointer to the model data
         * @return
         */
        std::unique_ptr<llama_sampler> into_llama_sampler(const llama_model *pModel) const;
    };

    /**
     *
     */
    struct generation_params_t {
        uint32_t max_new_tokens = std::numeric_limits<uint32_t>::max();
    };

    struct generation_context_t {
        generation_params_t generation_params;
        sampling_params_t sampling_params;
        std::span<const llama_token> input_tokens;
        std::span<llama_token> generated_tokens;
    };

    /**
     *
     */
    class worker_t {
    private:
        const std::shared_ptr<llama_model> mModel_;
        const llama_context_params mParams_;

    public:
        /**
         *
         * @param model
         * @param params
         */
        worker_t(std::shared_ptr<llama_model> model, const llama_context_params &params);

        /**
         *
         * @param context
         * @param generation_context
         * @param callback
         */
        size_t
        generate(llama_context *, const generation_context_t &, const std::optional<llama_decode_callback> &) const;

        /**
         *
         */
        void loop(std::stop_source &driver, std::queue<generation_context_t> &backlog) const;
    };


    class backend_base_t {

    protected:
        std::shared_ptr<llama_model> mModel_;

    public:

        /**
         *
         * @param model
         */
        explicit backend_base_t(llama_model *model);

        /**
         * Destructor
         */
        ~backend_base_t();

        /**
         *
         * @param tokens
         * @params out
         * @param params
         * @param maxNewTokens
         * @return
         */
        [[nodiscard("Generated tokens will be freed after this call if not assigned to an lvalue")]]
        virtual std::expected<size_t, backend_error_t> generate(
                std::span<const llama_token> input_tokens,
                std::span<llama_token> generated_tokens,
                const generation_params_t &generation_params,
                const sampling_params_t &sampling_params,
                const std::optional<llama_decode_callback> &callback
        ) = 0;

        /**
         *
         * @param tokens
         * @param params
         * @param maxNewTokens
         * @return
         */
        [[nodiscard("Generated tokens will be freed after this call if not assigned to an lvalue")]]
        std::expected<std::vector<llama_token>, backend_error_t> generate(
                std::span<const llama_token> tokens,
                const generation_params_t &generation_params,
                const sampling_params_t &sampling_params,
                const std::optional<llama_decode_callback> &callback = std::nullopt
        );
    };


    class single_worker_backend_t : backend_base_t {
    private:
        constexpr const static auto llama_context_factory = [](llama_model *pModel) -> llama_context_smart_ptr {
            auto llParams = llama_context_default_params();
            llParams.flash_attn = true;
            llParams.n_batch = 1;
            llParams.no_perf = true;
            llParams.attention_type = llama_attention_type::LLAMA_ATTENTION_TYPE_CAUSAL;

            return {llama_new_context_with_model(pModel, llParams), llama_context_deleter};
        };

        llama_context_smart_ptr mContext_;
        worker_t mWorker_;

    public:
        explicit single_worker_backend_t(llama_model *pModel, const std::optional<llama_context_params> &);

        using backend_base_t::generate;

        std::expected<size_t, backend_error_t>
        generate(
                std::span<const llama_token> tokens,
                std::span<llama_token> out,
                const generation_params_t &generation_params,
                const sampling_params_t &sampling_params,
                const std::optional<llama_decode_callback> &callback
        ) override;
    };

    class multi_worker_backend_t : backend_base_t {
    private:
        llama_context_smart_ptr mContext_;

    public:
        std::expected<size_t, backend_error_t> generate(
                std::span<const llama_token>,
                std::span<llama_token>,
                const generation_params_t &generation_params,
                const sampling_params_t &sampling_params,
                const std::optional<llama_decode_callback> &callback
        ) override;
    };
}

#endif //TGI_LLAMA_CPP_BACKEND_BACKEND_HPP
