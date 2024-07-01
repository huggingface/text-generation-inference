cmake_minimum_required(VERSION 3.26)

project(tgi-trtllm-backend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

option(TGI_TRTLLM_BACKEND_BUILD_TESTS "Enable building the unittests suite" OFF)


#### External dependencies ####
# Logging library (SPDLOG)
set(SPDLOG_USE_FMT ON)
fetchcontent_declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v2.x
)
fetchcontent_makeavailable(spdlog)

# TensorRT-LLM
fetchcontent_declare(
        trtllm
        GIT_REPOSITORY https://github.com/nvidia/tensorrt-llm.git
        GIT_TAG 9691e12bce7ae1c126c435a049eb516eb119486c
)
fetchcontent_populate(trtllm)
include_directories("${trtllm_SOURCE_DIR}/cpp/include")
message(STATUS "Found TensorRT-LLM: ${trtllm_SOURCE_DIR}")

# TGI TRTLLM Backend definition
add_library(tgi_trtllm_backend_impl STATIC include/backend.h lib/backend.cpp)
target_include_directories(tgi_trtllm_backend_impl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tgi_trtllm_backend_impl PUBLIC spdlog::spdlog)

#### Unit Tests ####
if (${TGI_TRTLLM_BACKEND_BUILD_TESTS})
    message(STATUS "Building tests")
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2
            GIT_TAG v3.6.0
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(tgi_trtllm_backend_tests)
    target_link_libraries(tests PRIVATE Catch2::Catch2::Catch2WithMain)

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    catch_discover_tests(tests)
endif ()