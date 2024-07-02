cmake_minimum_required(VERSION 3.20)

project(tgi-trtllm-backend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
include(ExternalProject)

option(TGI_TRTLLM_BACKEND_BUILD_TESTS "Enable building the unittests suite" OFF)
set(TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST "86-real;89-real;90-real" CACHE STRING "List of CUDA architectures to support")
set(TGI_TRTLLM_BACKEND_TRT_ROOT "/usr/local/tensorrt" CACHE PATH "Path where TensorRT libraries and headers are located")
set(TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/include" CACHE PATH "Path where TensorRT headers are located")
set(TGI_TRTLLM_BACKEND_TRT_LIB_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/lib" CACHE PATH "Path where TensorRT libraries are located")

if (NOT EXISTS ${TGI_TRTLLM_BACKEND_TRT_ROOT})
    message(FATAL_ERROR "TensorRT specified location: ${TGI_TRTLLM_BACKEND_TRT_ROOT} doesn't exist")
else ()
    if (NOT EXISTS ${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR})
        message(FATAL_ERROR "TensorRT headers were not found at: ${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR}")
    endif ()

    if (NOT EXISTS ${TGI_TRTLLM_BACKEND_TRT_LIB_DIR})
        message(FATAL_ERROR "TensorRT libraries were not found at: ${TGI_TRTLLM_BACKEND_TRT_LIB_DIR}")
    endif ()
endif ()

#### External dependencies ####
include(cmake/spdlog.cmake)
include(cmake/trtllm.cmake)

# TGI TRTLLM Backend definition
add_library(tgi_trtllm_backend_impl STATIC include/backend.h lib/backend.cpp)
target_include_directories(tgi_trtllm_backend_impl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tgi_trtllm_backend_impl PUBLIC spdlog::spdlog tensorrt_llm nvinfer_plugin_tensorrt_llm)

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