//
// Created by mfuntowicz on 12/3/24.
//

#include <catch2/catch_all.hpp>
#include <nlohmann/json.hpp>
#include <tensorrt_llm/executor/executor.h>

#include "backend.hpp"

using namespace huggingface::tgi::backends::trtllm;

TEST_CASE("parse generation_config.json all set", "[generation_config_t]")
{
    const json config_j = {{"temperature",  0.6},
                           {"top_p",        0.95},
                           {"eos_token_id", {1, 2, 3}}};
    const auto generation_config = generation_config_t(config_j);

    REQUIRE_THAT(generation_config.temperature, Catch::Matchers::WithinAbs(0.6, 1e-6));
    REQUIRE_THAT(generation_config.top_p, Catch::Matchers::WithinAbs(0.95, 1e-6));

    // Stop words
    REQUIRE_FALSE(generation_config.stop_words.empty());
    REQUIRE(generation_config.stop_words.size() == config_j["/eos_token_id"_json_pointer].size());

    for (auto [lhs, rhs]: std::views::zip(generation_config.stop_words, std::list<std::vector<int32_t>>{{1},
                                                                                                        {2},
                                                                                                        {3}})) {
        // Currently we do not support multi-tokens stop words
        REQUIRE(lhs.size() == 1);
        REQUIRE(rhs.size() == 1);
        REQUIRE_THAT(lhs, Catch::Matchers::UnorderedEquals(rhs));
    }
}

TEST_CASE("parse generation_config.json default", "[generation_config_t]")
{
    const json config_j = {{"eos_token_id", {1, 2, 3}}};
    const auto generation_config = generation_config_t(config_j);

    REQUIRE_THAT(generation_config.temperature, Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(generation_config.top_p, Catch::Matchers::WithinAbs(1.0, 1e-6));

    REQUIRE_FALSE(generation_config.stop_words.empty());
    REQUIRE(generation_config.stop_words.size() == config_j["/eos_token_id"_json_pointer].size());

    for (auto [lhs, rhs]: std::views::zip(generation_config.stop_words, std::list<std::vector<int32_t>>{{1},
                                                                                                        {2},
                                                                                                        {3}})) {
        // Currently we do not support multi-tokens stop words
        REQUIRE(lhs.size() == 1);
        REQUIRE(rhs.size() == 1);
        REQUIRE_THAT(lhs, Catch::Matchers::UnorderedEquals(rhs));
    }
}

TEST_CASE("parse generation_config.json empty", "[generation_config_t]")
{
    const json config_j = {{"eos_token_id", {}}};
    const auto generation_config = generation_config_t(config_j);

    REQUIRE_THAT(generation_config.temperature, Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(generation_config.top_p, Catch::Matchers::WithinAbs(1.0, 1e-6));

    REQUIRE(generation_config.stop_words.empty());

    const json config_j2 = {};
    const auto generation_config2 = generation_config_t(config_j);

    REQUIRE_THAT(generation_config2.temperature, Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(generation_config2.top_p, Catch::Matchers::WithinAbs(1.0, 1e-6));

    REQUIRE(generation_config2.stop_words.empty());
}

TEST_CASE("parallel_config single", "[backend_workspace_t]")
{
    // Generate temporary folder
    const auto tmp_p = std::filesystem::temp_directory_path();
    const auto config_p = tmp_p / "config.json";
    const auto generation_config_p = tmp_p / "generation_config.json";

    // Generate content
    std::ofstream o_config(config_p);
    o_config << R"({"pretrained_config": {"mapping": {"world_size": 2}}})"_json;
    o_config.close();

    std::ofstream o_generation_config(generation_config_p);
    o_generation_config << R"({"eos_token_id": []})"_json;
    o_generation_config.close();

    const auto workspace = backend_workspace_t(tmp_p.generic_string(), tmp_p.generic_string());
    const auto parallel = workspace.parallel_config();
    REQUIRE(parallel.getCommunicationMode() == tle::CommunicationMode::kORCHESTRATOR);
    REQUIRE(parallel.getCommunicationType() == tle::CommunicationType::kMPI);

    std::filesystem::remove(config_p);
    std::filesystem::remove(generation_config_p);
}

TEST_CASE("parallel_config multi", "[backend_workspace_t]")
{
    // Generate temporary folder
    const auto tmp_p = std::filesystem::temp_directory_path();
    const auto config_p = tmp_p / "config.json";
    const auto generation_config_p = tmp_p / "generation_config.json";

    // Generate content
    std::ofstream o_config(config_p);
    o_config << R"({"pretrained_config": {"mapping": {"world_size": 1}}})"_json;
    o_config.close();

    std::ofstream o_generation_config(generation_config_p);
    o_generation_config << R"({"eos_token_id": []})"_json;
    o_generation_config.close();

    const auto workspace = backend_workspace_t(tmp_p.generic_string(), tmp_p.generic_string());
    const auto parallel = workspace.parallel_config();
    REQUIRE(parallel.getCommunicationMode() == tle::CommunicationMode::kLEADER);
    REQUIRE(parallel.getCommunicationType() == tle::CommunicationType::kMPI);

    std::filesystem::remove(config_p);
    std::filesystem::remove(generation_config_p);
}

TEST_CASE("executor_config", "[backend_workspace_t]")
{

}

TEST_CASE("sampling_params_t to tle::SamplingConfig", "[backend_t]")
{
    const sampling_params_t params = {40, 0.95, 0.9, 1.0, 0.6, 2014};
    const auto config = static_cast<tle::SamplingConfig>(params);

    REQUIRE(config.getTopK().has_value());
    REQUIRE(config.getTopK().value() == params.top_k);

    REQUIRE(config.getSeed().has_value());
    REQUIRE(config.getSeed().value() == params.seed);

    REQUIRE(config.getTopP().has_value());
    REQUIRE_THAT(*config.getTopP(), Catch::Matchers::WithinAbs(params.top_p, 1e-6f));

    REQUIRE(config.getRepetitionPenalty().has_value());
    REQUIRE_THAT(*config.getRepetitionPenalty(), Catch::Matchers::WithinAbs(params.repetition_penalty, 1e-6f));

    REQUIRE(config.getFrequencyPenalty().has_value());
    REQUIRE_THAT(*config.getFrequencyPenalty(), Catch::Matchers::WithinAbs(params.frequency_penalty, 1e-6f));

    REQUIRE(config.getTemperature().has_value());
    REQUIRE_THAT(*config.getTemperature(), Catch::Matchers::WithinAbs(params.temperature, 1e-6f));
}
