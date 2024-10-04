//
// Created by mfuntowicz on 10/3/24.
//

#include <string_view>
#include <fmt/format.h>
#include <fmt/color.h>
#include <spdlog/spdlog.h>
#include "../csrc/backend.hpp"

int main(int argc, char** argv) {
    if(argc < 2) {
        fmt::print("No model folder provider");
        return 1;
    }

    spdlog::set_level(spdlog::level::debug);

    const std::string_view model_root = argv[1];
    auto backend = huggingface::tgi::backends::llama::CreateLlamaCppBackend(model_root);
    fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "Successfully initialized llama.cpp model from {}\n", model_root);
}