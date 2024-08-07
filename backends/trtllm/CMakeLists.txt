cmake_minimum_required(VERSION 3.20)

project(tgi-trtllm-backend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
include(ExternalProject)

option(TGI_TRTLLM_BACKEND_BUILD_TESTS "Enable building the unittests suite" OFF)
option(TGI_TRTLLM_BACKEND_BUILD_EXAMPLES "Enable building the examples suite" OFF)
set(TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST "89-real" CACHE STRING "List of CUDA architectures to support")
set(TGI_TRTLLM_BACKEND_TRT_ROOT "/usr/local/tensorrt" CACHE STRING "Path where TensorRT libraries and headers are located")
set(TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/include" CACHE STRING "Path where TensorRT headers are located")
set(TGI_TRTLLM_BACKEND_TRT_LIB_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/lib" CACHE STRING "Path where TensorRT libraries are located")

# We are using nvidia-ml to query at runtime device information to enable some architecture-specific features
find_package(CUDAToolkit 12.5 REQUIRED COMPONENTS CUDA::cudart CUDA::nvml)

#### External dependencies ####
include(cmake/fmt.cmake)
include(cmake/json.cmake)
include(cmake/spdlog.cmake)
include(cmake/trtllm.cmake)

# Let's build TRTLLM as part of CMake
add_subdirectory("${trtllm_SOURCE_DIR}/cpp" "${trtllm_SOURCE_DIR}/..")

# Tell CMake to need try to override the RPATH for executorWorker as it has not information on how to do so
set_target_properties(executorWorker PROPERTIES SKIP_BUILD_RPATH TRUE)

# TGI TRTLLM Backend definition
add_library(tgi_trtllm_backend_impl STATIC include/backend.h lib/backend.cpp include/hardware.h)
include_directories(${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR})
target_include_directories(tgi_trtllm_backend_impl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_include_directories(tgi_trtllm_backend_impl PUBLIC "${trtllm_SOURCE_DIR}/cpp/include")
target_link_libraries(tgi_trtllm_backend_impl PRIVATE tensorrt_llm nvinfer_plugin_tensorrt_llm tensorrt_llm_nvrtc_wrapper CUDA::cudart CUDA::nvml)
target_link_libraries(tgi_trtllm_backend_impl PUBLIC nlohmann_json::nlohmann_json spdlog::spdlog fmt::fmt)

# This install all the artifacts in CMAKE_INSTALL_PREFIX under include/ lib/ bin/ to make easy to link / find it back
install(TARGETS tgi_trtllm_backend_impl tensorrt_llm nvinfer_plugin_tensorrt_llm decoder_attention executorWorker)
install(FILES ${TRTLLM_NVRTC_WRAPPER_LIBRARY_PATH} ${TRTLLM_EXECUTOR_STATIC_LIBRARY_PATH} TYPE LIB)

#### Unit Tests ####
if (${TGI_TRTLLM_BACKEND_BUILD_TESTS})
    message(STATUS "Building tests")
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2
            GIT_TAG v3.6.0
    )
    FetchContent_MakeAvailable(Catch2)

    #    add_executable(tgi_trtllm_backend_tests tests/infer_test.cpp)
    #    target_link_libraries(tgi_trtllm_backend_tests PRIVATE tgi_trtllm_backend_impl Catch2::Catch2WithMain nlohmann_json::nlohmann_json spdlog::spdlog fmt::fmt CUDA::cudart CUDA::nvml)

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    #    catch_discover_tests(tgi_trtllm_backend_tests)
endif ()
