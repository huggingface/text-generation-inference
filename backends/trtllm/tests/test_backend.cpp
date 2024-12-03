//
// Created by mfuntowicz on 12/3/24.
//

#include <catch2/catch_all.hpp>
#include <nlohmann/json.hpp>
#include "../csrc/backend.hpp"

using namespace huggingface::tgi::backends::trtllm;

TEST_CASE("parse generation_config.json", "[generation_config_t]")
{
    const json config_j = {{"temperature", 0.6}, {"top_p", 0.95}, {"eos_token_id", {1,2,3}}};
    const auto generation_config = generation_config_t(config_j);

    REQUIRE_FALSE(generation_config.stop_words.empty());
    REQUIRE(generation_config.stop_words.size() == config_j["/eos_token_id"_json_pointer].size());

    for (auto [lhs, rhs] : std::views::zip(generation_config.stop_words, std::list<std::vector<int32_t>>{{1}, {2}, {3}}))
    {
        // Currently we do not support multi-tokens stop words
        REQUIRE(lhs.size() == 1);
        REQUIRE(rhs.size() == 1);
        REQUIRE_THAT(lhs, Catch::Matchers::UnorderedEquals(rhs));
    }
}

TEST_CASE("parallel_config", "[backend_workspace_t]")
{

}

TEST_CASE("executor_config", "[backend_workspace_t]")
{

}