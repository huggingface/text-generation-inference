//
// Created by mfuntowicz on 10/3/24.
//
#include <memory>

#include <llama.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#include "../csrc/backend.hpp"

using namespace huggingface::tgi::backends::llamacpp;

const auto llama_model_deleter = [](llama_model *model) { llama_free_model(model); };

int main(int argc, char **argv) {
    if (argc < 2) {
        fmt::print("No model folder provider");
        return 1;
    }

    spdlog::set_level(spdlog::level::debug);

    const auto modelPath = absolute(std::filesystem::path(argv[1]));
    const auto params = llama_model_default_params();
    auto model = std::shared_ptr<llama_model>(
            llama_load_model_from_file(modelPath.c_str(), params),
            llama_model_deleter
    );

    auto prompt = std::string("My name is Morgan");
    auto tokens = std::vector<llama_token>(128);
    const auto nb_tokens = llama_tokenize(model.get(), prompt.c_str(), prompt.size(), tokens.data(), tokens.size(),
                                          true,
                                          false);
    tokens.resize(nb_tokens);
    llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISTRIBUTE);
    auto backend = worker_t(model, llama_context_default_params());

    fmt::println("Tokenized: {}", tokens);

    // generate
    auto generated_tokens = std::vector<llama_token>(32);
    const auto n_generated_tokens = backend.generate(
            {{.max_new_tokens = 32}, {.top_k = 40, .top_p = 0.95, .temperature = 0.8},
             tokens},
            [&generated_tokens](llama_token new_token_id, float_t logit, bool is_eos, size_t step) -> bool {
                generated_tokens.emplace(generated_tokens.begin() + (step - 1), new_token_id);
                return false;
            }
    );
    generated_tokens.resize(n_generated_tokens.value());

    std::string decoded = std::string(256, 'a');
    const size_t length = llama_detokenize(model.get(),
                                           generated_tokens.data(),
                                           generated_tokens.size(),
                                           decoded.data(),
                                           decoded.size(),
                                           false, false);
    decoded.resize(std::min(length, decoded.size()));
    fmt::println("Generated tokens: {}", generated_tokens);
    fmt::println("Generated text:   {}", decoded);
}
