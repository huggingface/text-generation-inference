#ifndef TGI_BACKEND_TRTLLM
#define TGI_BACKEND_TRTLLM

#include <cmath>
#include <cstdint>
#include <expected>
#include <fstream>
#include <list>
#include <span>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <tensorrt_llm/executor/executor.h>

namespace huggingface::tgi::backends::trtllm {
    namespace tle = tensorrt_llm::executor;
    using json = nlohmann::json;
    using request_id_t = uint64_t;
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
        float_t temperature;
        uint64_t seed;

        constexpr explicit operator tle::SamplingConfig() const {
            return tle::SamplingConfig{
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
                    std::nullopt
            };
        }
    };

    /**
     * Represent possible values from transformers generation `generation_config.json`.
     * It usually stores default sampling parameters to use, such as top_p, temperature, etc.
     */
    struct generation_config_t {
        float_t top_p;
        float_t temperature;
        std::list<std::vector<int32_t>> stop_words;

        constexpr explicit generation_config_t(const json &config) :
                top_p(config.value("top_p", 1.0f)), temperature(config.value("temperature", 1.0f)), stop_words(0) {
            if (config.contains("/eos_token_id"_json_pointer) && config["/eos_token_id"_json_pointer].is_array()) {
                const auto &eos_token_id = config["/eos_token_id"_json_pointer];
                std::for_each(eos_token_id.begin(), eos_token_id.end(), [this](const auto token_id) {
                    stop_words.emplace_back(1, token_id.template get<int32_t>());
                });

                SPDLOG_DEBUG("Detected {:d} predefined stop_words from generation_config.json", stop_words.size());
            }
        }
    };

    /**
     * Helper class representing various items which are stored within the TensorRT-LLM engines folder and
     * can be retrieved at runtime
     */
    class backend_workspace_t {
    private:
        constexpr static auto as_json = [](const std::filesystem::path &path) -> json {
            std::ifstream config_f(path);
            return json::parse(config_f);
        };

        std::filesystem::path engines_folder_;
        std::filesystem::path executor_worker_path_;
        json config_;
        generation_config_t generation_config_;

    public:
        backend_workspace_t(std::filesystem::path &engines_folder, std::filesystem::path &executor_worker_path) :
                engines_folder_(engines_folder),
                executor_worker_path_(executor_worker_path),
                config_(as_json(engines_folder / "config.json")),
                generation_config_(as_json(engines_folder / "generation_config.json")) {};

        backend_workspace_t(std::filesystem::path &&engines_folder, std::filesystem::path &&executor_worker_path) :
                engines_folder_(engines_folder),
                executor_worker_path_(executor_worker_path),
                config_(as_json(engines_folder / "config.json")),
                generation_config_(as_json(engines_folder / "generation_config.json")) {};

        /**
         * Path to the folder containing the TensorRT-LLM engines
         * @return local filesystem path to the folder
         */
        [[nodiscard]] constexpr std::filesystem::path engines_folder() const { return engines_folder_; }

        /**
         * Hugging Face transformers' generated `generation_config_t` mapping information stored in the
         * `generation_config.json` holding default generation parameters.
         * @return `generation_config_t`
         */
        [[nodiscard]] constexpr const generation_config_t &generation_config() const { return generation_config_; }

        /**
         * Factory method returning new `tensorrt_llm::executor::ParallelConfig` instance used
         * to initialize `tensorrt_llm::executor::Executor` with multi-instance communication information
         * @return `tensorrt_llm::executor::ParallelConfig` instance
         */
        [[nodiscard]] tle::ParallelConfig parallel_config() const;

        /**
         * Factory method returning new `tensorrt_llm::executor::ExecutorConfig` instance used
         * to initialize `tensorrt_llm::executor::Executor`
         * @return `tensorrt_llm::executor::ExecutorConfig` instance
         */
        [[nodiscard]] tle::ExecutorConfig executor_config() const;
    };

    /**
     * Error raised by the underlying backend implementation
     */
    enum backend_error_t {
        EXECUTOR_NOT_READY = 3,
        EXECUTOR_SCHEDULING_FAILED = 4,
    };


    /**
     * Actual TensorRT-LLM backend implementation interacting with TensorRT-LLM Executor service to
     * - schedule new request
     * - pull status of submitted request(s)
     * - cancel submitted request(s)
     */
    class backend_t {
    private:
        backend_workspace_t workspace;
        tle::Executor executor_;

    public:
        backend_t(std::filesystem::path &engines_folder, std::filesystem::path &executor_worker_path);

        backend_t(std::filesystem::path &&engines_folder, std::filesystem::path &&executor_worker_path)
                : backend_t(engines_folder, executor_worker_path) {};

        /**
         * Submit a new request to the executor
         * @param token_ids
         * @param generation_params
         * @param sampling_params
         * @return Either newly submitted request's id or the error why it failed to submit
         */
        [[nodiscard("Discarded executor request_id needs to be assigned")]]
        std::expected<request_id_t, backend_error_t>
        submit(std::span<const token_id_t> token_ids, generation_params_t generation_params,
               sampling_params_t sampling_params) noexcept;

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

    /**
     * Create a TensorRT-LLM executor from a workspace
     */
    const auto executor_factory_initializer = [](const backend_workspace_t &workspace) -> tle::Executor {
        return {workspace.engines_folder(), tensorrt_llm::executor::ModelType::kDECODER_ONLY,
                workspace.executor_config()};
    };
}

/**
 * Helper structures to define formatting strategies for various types in the backend
 */
template<>
struct fmt::formatter<huggingface::tgi::backends::trtllm::generation_params_t> : formatter<string_view> {
    auto format(huggingface::tgi::backends::trtllm::generation_params_t const &c,
                format_context &ctx) const -> format_context::iterator {
        return fmt::format_to(ctx.out(), "generation_params_t{{ max_new_tokens={:d} }}", c.max_new_tokens);
    }
};

template<>
struct fmt::formatter<huggingface::tgi::backends::trtllm::sampling_params_t> : formatter<string_view> {
    auto format(huggingface::tgi::backends::trtllm::sampling_params_t const &c,
                format_context &ctx) const -> format_context::iterator {
        return fmt::format_to(
                ctx.out(),
                "sampling_params_t{{ top_k={:d}, top_p={:.3f}, repetition_penalty={:.3f}, frequency_penalty={:.3f}, temperature={:.3f}, seed={:d} }}",
                c.top_k, c.top_p, c.repetition_penalty, c.frequency_penalty, c.temperature, c.seed
        );
    }
};

#endif
