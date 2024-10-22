//
// Created by mfuntowicz on 10/3/24.
//

#include <string_view>
#include <fmt/format.h>
#include <fmt/std.h>
#include <fmt/color.h>
#include <spdlog/spdlog.h>
#include "../csrc/backend.hpp"

int main(int argc, char** argv) {
    if(argc < 2) {
        fmt::print("No model folder provider");
        return 1;
    }

    spdlog::set_level(spdlog::level::debug);

    const auto modelPath = absolute(std::filesystem::path(argv[1]));
    if(auto backend = huggingface::tgi::backends::llama::CreateLlamaCppBackend(modelPath); backend.has_value())
        fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "Successfully initialized llama.cpp model from {}\n", modelPath);
}