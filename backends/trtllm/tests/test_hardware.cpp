//
// Created by mfuntowicz on 11/16/24.
//

#include <catch2/catch_all.hpp>
#include "../csrc/hardware.hpp"

using namespace huggingface::tgi::hardware::cuda;

TEST_CASE("is_at_least_<arch>") {
    const static auto VOLTA_CAPABILITIES = compute_capabilities_t(7, 0);
    REQUIRE(VOLTA_CAPABILITIES.is_at_least_volta());
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least_turing());
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least_ampere());
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least_ada_lovelace());
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least_hopper());

    const static auto TURING_CAPABILITIES = compute_capabilities_t(7, 5);
    REQUIRE(TURING_CAPABILITIES.is_at_least_volta());
    REQUIRE(TURING_CAPABILITIES.is_at_least_turing());
    REQUIRE_FALSE(TURING_CAPABILITIES.is_at_least_ampere());
    REQUIRE_FALSE(TURING_CAPABILITIES.is_at_least_ada_lovelace());
    REQUIRE_FALSE(TURING_CAPABILITIES.is_at_least_hopper());

    const static auto AMPERE_CAPABILITIES = compute_capabilities_t(8, 0);
    REQUIRE(AMPERE_CAPABILITIES.is_at_least_volta());
    REQUIRE(AMPERE_CAPABILITIES.is_at_least_turing());
    REQUIRE(AMPERE_CAPABILITIES.is_at_least_ampere());
    REQUIRE_FALSE(AMPERE_CAPABILITIES.is_at_least_ada_lovelace());
    REQUIRE_FALSE(AMPERE_CAPABILITIES.is_at_least_hopper());

    const static auto ADA_LOVELACE_CAPABILITIES = compute_capabilities_t(8, 9);
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least_volta());
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least_turing());
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least_ampere());
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least_ada_lovelace());
    REQUIRE_FALSE(ADA_LOVELACE_CAPABILITIES.is_at_least_hopper());

    const static auto HOPPER_CAPABILITIES = compute_capabilities_t(9, 0);
    REQUIRE(HOPPER_CAPABILITIES.is_at_least_volta());
    REQUIRE(HOPPER_CAPABILITIES.is_at_least_turing());
    REQUIRE(HOPPER_CAPABILITIES.is_at_least_ampere());
    REQUIRE(HOPPER_CAPABILITIES.is_at_least_ada_lovelace());
    REQUIRE(HOPPER_CAPABILITIES.is_at_least_hopper());
}

TEST_CASE("is_at_least") {
    const static auto VOLTA_CAPABILITIES = compute_capabilities_t(7, 0);
    REQUIRE(VOLTA_CAPABILITIES.is_at_least(VOLTA));
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least(TURING));
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least(AMPERE));
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least(ADA_LOVELACE));
    REQUIRE_FALSE(VOLTA_CAPABILITIES.is_at_least(HOPPER));

    const static auto TURING_CAPABILITIES = compute_capabilities_t(7, 5);
    REQUIRE(TURING_CAPABILITIES.is_at_least(VOLTA));
    REQUIRE(TURING_CAPABILITIES.is_at_least(TURING));
    REQUIRE_FALSE(TURING_CAPABILITIES.is_at_least(AMPERE));
    REQUIRE_FALSE(TURING_CAPABILITIES.is_at_least(ADA_LOVELACE));
    REQUIRE_FALSE(TURING_CAPABILITIES.is_at_least(HOPPER));

    const static auto AMPERE_CAPABILITIES = compute_capabilities_t(8, 0);
    REQUIRE(AMPERE_CAPABILITIES.is_at_least(VOLTA));
    REQUIRE(AMPERE_CAPABILITIES.is_at_least(TURING));
    REQUIRE(AMPERE_CAPABILITIES.is_at_least(AMPERE));
    REQUIRE_FALSE(AMPERE_CAPABILITIES.is_at_least(ADA_LOVELACE));
    REQUIRE_FALSE(AMPERE_CAPABILITIES.is_at_least(HOPPER));

    const static auto ADA_LOVELACE_CAPABILITIES = compute_capabilities_t(8, 9);
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least(VOLTA));
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least(TURING));
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least(AMPERE));
    REQUIRE(ADA_LOVELACE_CAPABILITIES.is_at_least(ADA_LOVELACE));
    REQUIRE_FALSE(ADA_LOVELACE_CAPABILITIES.is_at_least(HOPPER));

    const static auto HOPPER_CAPABILITIES = compute_capabilities_t (9, 0);
    REQUIRE(HOPPER_CAPABILITIES.is_at_least(VOLTA));
    REQUIRE(HOPPER_CAPABILITIES.is_at_least(TURING));
    REQUIRE(HOPPER_CAPABILITIES.is_at_least(AMPERE));
    REQUIRE(HOPPER_CAPABILITIES.is_at_least(ADA_LOVELACE));
    REQUIRE(HOPPER_CAPABILITIES.is_at_least(HOPPER));
}
