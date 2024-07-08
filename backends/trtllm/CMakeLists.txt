cmake_minimum_required(VERSION 3.20)

project(tgi-trtllm-backend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
include(ExternalProject)

option(TGI_TRTLLM_BACKEND_BUILD_TESTS "Enable building the unittests suite" OFF)
set(TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST "89-real" CACHE STRING "List of CUDA architectures to support")
set(TGI_TRTLLM_BACKEND_TRT_ROOT "/usr/local/tensorrt" CACHE PATH "Path where TensorRT libraries and headers are located")
set(TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/include" CACHE PATH "Path where TensorRT headers are located")
set(TGI_TRTLLM_BACKEND_TRT_LIB_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/lib" CACHE PATH "Path where TensorRT libraries are located")

#### External dependencies ####
include(cmake/json.cmake)
include(cmake/spdlog.cmake)
include(cmake/trtllm.cmake)

find_package(CUDAToolkit REQUIRED COMPONENTS CUDA::nvml)

# TGI TRTLLM Backend definition
add_library(tgi_trtllm_backend_impl STATIC include/backend.h lib/backend.cpp)
target_include_directories(tgi_trtllm_backend_impl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
include_directories(${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR})
target_link_libraries(tgi_trtllm_backend_impl PUBLIC tensorrt_llm nvinfer_plugin_tensorrt_llm)
target_link_libraries(tgi_trtllm_backend_impl PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog CUDA::nvml)


#### Unit Tests ####
if (${TGI_TRTLLM_BACKEND_BUILD_TESTS})
    message(STATUS "Building tests")
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2
            GIT_TAG v3.6.0
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(tgi_trtllm_backend_tests tests/infer_test.cpp)
    target_link_libraries(tgi_trtllm_backend_tests PRIVATE tgi_trtllm_backend_impl Catch2::Catch2WithMain nlohmann_json::nlohmann_json spdlog::spdlog)

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    catch_discover_tests(tgi_trtllm_backend_tests)
endif ()