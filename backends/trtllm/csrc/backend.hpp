#include <cmath>
#include <cstdint>
#include <exception>
#include <expected>
#include <list>
#include <span>

#include <tensorrt_llm/executor/executor.h>

namespace huggingface::tgi::backends::trtllm {
    namespace tle = tensorrt_llm::executor;

    using request_id_t = uint32_t;
    using token_id_t = tle::TokenIdType;

    /**
     * Represent the parameters used for generation
     */
    struct generation_params_t {
        uint32_t max_new_tokens;
    };

    /**
     * Represent the parameters used to sample tokens from the logit distribution
     */
    struct sampling_params_t {
        uint32_t top_k;
        float_t top_p;
        float_t repetition_penalty;
        float_t frequency_penalty;
        float_t length_penalty;
        float_t temperature;
        uint64_t seed;

        explicit operator tle::SamplingConfig() const {
            return tle::SamplingConfig {
                1,
                top_k,
                top_p,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                seed,
                temperature,
                std::nullopt,
                std::nullopt,
                repetition_penalty,
                std::nullopt,
                frequency_penalty,
                length_penalty
            };
        }
    };

    /**
     *
     */
    class backend_exception_t: std::exception  {};

    /**
     *
     */
    class backend_t {
    private:
        tle::Executor executor_;
        std::list<std::vector<int32_t>> stop_words_;

    public:
        /**
         * Submit a new request to the executor
         * @param token_ids
         * @param generation_params
         * @param sampling_params
         * @return Either newly submitted request's id or the error why it failed to submit
         */
        [[nodiscard("Discarded executor request_id needs to be assigned")]]
        std::expected<request_id_t, backend_exception_t>
        submit(std::span<token_id_t> token_ids, generation_params_t generation_params, sampling_params_t sampling_params) noexcept;

        /**
         * Query the number of tokens available across all in-flight generations
         * @return
         */
        [[nodiscard("Pulling out the number of tokens")]]
        size_t num_tokens_ready() const noexcept;

        /**
         * Pull out newly generated tokens from the executor
         * @return
         */
        [[nodiscard("")]]
        std::vector<tle::Response> pull_tokens() noexcept;

        /**
         * Cancel the specified request on the executor' set
         * @param request_id Request's Identifier to remove from the in-flight executor
         */
        void cancel(request_id_t) noexcept;
    };
}