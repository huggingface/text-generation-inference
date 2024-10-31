//
// Created by mfuntowicz on 10/3/24.
//

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include "../csrc/backend.hpp"

using namespace huggingface::tgi::backends::llamacpp;

int main(int argc, char **argv) {
    if (argc < 2) {
        fmt::print("No model folder provider");
        return 1;
    }

    spdlog::set_level(spdlog::level::debug);
    
    const auto modelPath = absolute(std::filesystem::path(argv[1]));
    const auto params = llama_model_default_params();
    auto *model = llama_load_model_from_file(modelPath.c_str(), params);

    auto backend = single_worker_backend_t(model, {});

    // generate
    const auto promptTokens = {128000, 5159, 836, 374, 23809, 11};
    const auto out = backend.generate(promptTokens, {.max_new_tokens = 32}, {.top_k = 40});

    if (out.has_value())
        fmt::print(FMT_STRING("Generated: {}"), *out);
    else {
        const auto err = out.error();
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Got an error: {:d}", static_cast<uint8_t>(err));
    }
}
