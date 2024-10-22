//
// Created by mfuntowicz on 10/3/24.
//

#include <string_view>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include "../csrc/backend.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print("No model folder provider");
        return 1;
    }

    spdlog::set_level(spdlog::level::debug);

    const auto prompt = "My name is Morgan";

    const auto modelPath = absolute(std::filesystem::path(argv[1]));
    if (auto maybeBackend = huggingface::tgi::backends::llama::CreateLlamaCppBackend(modelPath); maybeBackend.has_value()) {
        // Retrieve the backend
        const auto& backend = *maybeBackend;

        // Generate
        const auto promptTokens = backend->Tokenize(prompt);
        const auto out = backend->Generate(promptTokens, 30, 1.0, 32);
        fmt::print(FMT_STRING("Generated: {}"), out);
    } else {
        switch (maybeBackend.error()) {
            case huggingface::tgi::backends::llama::TgiLlamaCppBackendError::MODEL_FILE_DOESNT_EXIST:
                fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Specified file {} doesnt exist", modelPath);
                return maybeBackend.error();
        }
    }
}
