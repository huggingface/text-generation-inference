//
// Created by mfuntowicz on 10/3/24.
//
#include <memory>

#include <llama.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>s
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
    auto model = std::unique_ptr<llama_model, decltype(llama_model_deleter)>(
            llama_load_model_from_file(modelPath.c_str(), params)
    );

    auto prompt = "My name is Morgan";
    auto tokens = std::vector<llama_token>(16);
    const auto nb_tokens = llama_tokenize(model.get(), prompt, sizeof(prompt), tokens.data(), tokens.size(), true,
                                          false);
    tokens.resize(nb_tokens);
    auto backend = worker_t{std::move(model), {.n_batch = 1, .n_threads = 4}};

    fmt::println("Tokenized: {}", tokens);

    // generate
    auto generated_tokens = std::vector<llama_token>(32);
    const auto n_generated_tokens = backend.generate(
            {{.max_new_tokens = 32}, {.top_k = 40}, tokens},
            [&generated_tokens](llama_token new_token_id, float_t logit, bool is_eos, size_t step) -> bool {
                generated_tokens.emplace(generated_tokens.begin() + (step - 1), new_token_id);
                return false;
            }
    );
    generated_tokens.resize(n_generated_tokens.value());
    fmt::println("Generated {} tokens", generated_tokens);
}
