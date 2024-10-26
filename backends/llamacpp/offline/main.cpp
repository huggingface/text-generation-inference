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

    const auto prompt = "My name is Morgan";

    const auto modelPath = absolute(std::filesystem::path(argv[1]));
    if (auto maybeBackend = TgiLlamaCppBackend::FromGGUF(modelPath); maybeBackend.has_value()) {
        // Retrieve the backend
        auto [model, context] = *maybeBackend;
        auto backend = TgiLlamaCppBackend(model, context);

        // Generate
        const auto promptTokens = backend.Tokenize(prompt);
        const auto out = backend.Generate(promptTokens, 30, 1.0, 2.0, 0.0, 32);

        if (out.has_value())
            fmt::print(FMT_STRING("Generated: {}"), *out);
        else {
            const auto err = out.error();
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Got an error: {:d}", static_cast<uint8_t>(err));
        }

    } else {
        switch (maybeBackend.error()) {
            case TgiLlamaCppBackendError::MODEL_FILE_DOESNT_EXIST:
                fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Specified file {} doesnt exist", modelPath);
                return maybeBackend.error();
        }
    }
}
