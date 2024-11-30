#include <cmath>
#include <cstdint>
#include <exception>
#include <expected>
#include <fstream>
#include <list>
#include <span>

#include <nlohmann/json.hpp>
#include <spdlog/fmt/fmt.h>
#include <tensorrt_llm/executor/executor.h>

#include <hardware.hpp>

namespace huggingface::tgi::backends::trtllm {
    namespace tle = tensorrt_llm::executor;
    using json = nlohmann::json;
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

        constexpr explicit operator tle::SamplingConfig() const {
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
    struct generation_config_t {
        float_t top_p;
        float_t temperature;
        std::list<std::vector<int32_t>> stop_words;

        explicit generation_config_t(const json &config):
            top_p(config.value("top_p", 1.0f)), temperature( config.value("temperature", 1.0f)), stop_words(0) {
            if(config.contains("/eos_token_id"_json) && config["/eos_token_id"_json].is_array()) {
                const auto& eos_token_id = config["eos_token_id"];
                std::for_each(eos_token_id.begin(), eos_token_id.end(), [this](int32_t token_id) {
                    stop_words.push_back({token_id});
                });
            }
        }
    };

    /**
     *
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
        backend_workspace_t(std::filesystem::path &engines_folder, std::filesystem::path &executor_worker_path):
            engines_folder_(engines_folder),
            executor_worker_path_(executor_worker_path),
            config_(as_json(engines_folder / "config.json")),
            generation_config_(as_json(engines_folder / "generation_config.json")) {};

        backend_workspace_t(std::filesystem::path &&engines_folder, std::filesystem::path &&executor_worker_path):
                engines_folder_(engines_folder),
                executor_worker_path_(executor_worker_path),
                config_(as_json(engines_folder / "config.json")),
                generation_config_(as_json(engines_folder / "generation_config.json")) {};

        /**
         * Path to the folder containing the TensorRT-LLM engines
         * @return local filesystem path to the folder
         */
        [[nodiscard]] std::filesystem::path engines_folder() const { return engines_folder_; }

        /**
         *
         * @return
         */
        [[nodiscard]] const generation_config_t& generation_config() const { return generation_config_; }

        /**
         *
         * @return
         */
        [[nodiscard]] constexpr tle::ParallelConfig parallel_config() const;

        /**
         *
         * @return
         */
        [[nodiscard]] constexpr tle::ExecutorConfig executor_config() const;
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

    /**
     * Create a TensorRT-LLM executor from a workspace
     */
    const auto executor_factory_initializer = [](const backend_workspace_t &workspace) -> tle::Executor {
        return { workspace.engines_folder(), tensorrt_llm::executor::ModelType::kDECODER_ONLY, workspace.executor_config() };
    };
}

template <> struct fmt::formatter<huggingface::tgi::backends::trtllm::generation_params_t>: formatter<string_view> {
    auto format(huggingface::tgi::backends::trtllm::generation_params_t c, format_context& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "generation_params_t{{ max_new_tokens={:d} }}", c.max_new_tokens);
    }
};

template <> struct fmt::formatter<huggingface::tgi::backends::trtllm::sampling_params_t>: formatter<string_view> {
    auto format(huggingface::tgi::backends::trtllm::sampling_params_t c, format_context& ctx) const -> format_context::iterator {
        return format_to(
                ctx.out(),
                "sampling_params_t{{ top_k={:d}, top_p={:.3f}, repetition_penalty={:.3f}, frequency_penalty={:.3f}, length_penalty={:.3f}, temperature={:.3f}, seed={:d} }}",
                c.top_k, c.top_p, c.repetition_penalty, c.frequency_penalty, c.length_penalty, c.temperature, c.seed
        );
    }
};