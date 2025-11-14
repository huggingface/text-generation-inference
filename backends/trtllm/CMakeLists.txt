cmake_minimum_required(VERSION 3.20)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

project(tgi-trtllm-backend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
include(ExternalProject)
include(CheckCXXCompilerFlag)

option(TGI_TRTLLM_BACKEND_BUILD_TESTS "Enable building the unittests suite" OFF)
option(TGI_TRTLLM_BACKEND_BUILD_EXAMPLES "Enable building the examples suite" OFF)
option(TGI_TRTLLM_BACKEND_BUILD_USE_LLD "Enable lld linker instead of ld" OFF)
set(TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST "89-real" CACHE STRING "List of CUDA architectures to support")
set(TGI_TRTLLM_BACKEND_TRT_ROOT "/usr/local/tensorrt" CACHE STRING "Path where TensorRT libraries and headers are located")
set(TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/include" CACHE STRING "Path where TensorRT headers are located")
set(TGI_TRTLLM_BACKEND_TRT_LIB_DIR "${TGI_TRTLLM_BACKEND_TRT_ROOT}/lib" CACHE STRING "Path where TensorRT libraries are located")

# We are using nvidia-ml to query at runtime device information to enable some architecture-specific features
find_package(CUDAToolkit 12.6 REQUIRED COMPONENTS CUDA::cudart CUDA::nvml)
find_package(MPI REQUIRED)

#### External dependencies ####
include(cmake/json.cmake)
include(cmake/spdlog.cmake)
include(cmake/trtllm.cmake)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TGI_TRTLLM_BACKEND_DEBUG ON)
    add_compile_definitions(TGI_TRTLLM_BACKEND_DEBUG=1)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE)
endif ()

if (${TGI_TRTLLM_BACKEND_BUILD_USE_LLD})
    message(STATUS "Using lld linker")
    add_link_options("-fuse-ld=lld")
endif ()

# Let's build TRTLLM as part of CMake
add_subdirectory("${trtllm_SOURCE_DIR}/cpp" "${trtllm_SOURCE_DIR}/..")

# Tell CMake to need try to override the RPATH for executorWorker as it has not information on how to do so
set_target_properties(executorWorker PROPERTIES SKIP_BUILD_RPATH TRUE)

# TGI TRTLLM Backend definition
add_library(tgi_trtllm_backend_impl STATIC csrc/hardware.hpp csrc/backend.hpp csrc/backend.cpp)
include_directories(${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR})
target_include_directories(tgi_trtllm_backend_impl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>
        #        $<INSTALL_INTERFACE:csrc>
)
target_include_directories(tgi_trtllm_backend_impl PUBLIC "${trtllm_SOURCE_DIR}/cpp/include")
target_link_libraries(tgi_trtllm_backend_impl PRIVATE CUDA::cudart CUDA::nvml)
target_link_libraries(tgi_trtllm_backend_impl PUBLIC nlohmann_json::nlohmann_json spdlog::spdlog)
target_link_libraries(tgi_trtllm_backend_impl PRIVATE tensorrt_llm nvinfer_plugin_tensorrt_llm tensorrt_llm_nvrtc_wrapper)

# This install all the artifacts in CMAKE_INSTALL_PREFIX under include/ lib/ bin/ to make easy to link / find it back
install(TARGETS tgi_trtllm_backend_impl)
#install(TARGETS cutlass_src fb_gemm_src fpA_intB_gemm_src gemm_swiglu_sm90_src kernels_src)
install(TARGETS decoder_attention_0 decoder_attention_1)
install(TARGETS tensorrt_llm nvinfer_plugin_tensorrt_llm decoder_attention_src executorWorker)
install(FILES ${TRTLLM_NVRTC_WRAPPER_LIBRARY_PATH} TYPE LIB)
if (NOT ${TGI_TRTLLM_BACKEND_DEBUG})
    install(FILES ${TRTLLM_EXECUTOR_STATIC_LIBRARY_PATH} TYPE LIB)
endif ()


#### Unit Tests ####
if (${TGI_TRTLLM_BACKEND_BUILD_TESTS} AND CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Building tests")
    option(TGI_TRTLLM_BACKEND_ENABLE_ASAN "Enable AddressSanitizer")
    option(TGI_TRTLLM_BACKEND_ENABLE_UBSAN "Enable UndefinedSanitizer")

    FetchContent_Declare(
            Catch2
            URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    )
    FetchContent_MakeAvailable(Catch2)

    # This attempt to detect if the compiler can emit warning if it can't apply return value optimization from a function
    check_cxx_compiler_flag("-Wnrvo" COMPILER_SUPPORT_WARNING_ON_NVRO)
    if (${COMPILER_SUPPORT_WARNING_ON_NVRO})
        message(STATUS "Enabling non-NVRO detection")
        target_compile_options(tgi_trtllm_backend_impl PRIVATE -Wnrvo)
    endif ()
    target_compile_options(tgi_trtllm_backend_impl PRIVATE -Wall)

    cmake_path(GET TRTLLM_NVRTC_WRAPPER_LIBRARY_PATH PARENT_PATH TRTLLM_NVRTC_WRAPPER_PARENT_LIBRARY_PATH)
    message(STATUS "Adding linking path: ${TRTLLM_NVRTC_WRAPPER_PARENT_LIBRARY_PATH}")

    add_executable(tgi_trtllm_backend_tests tests/test_hardware.cpp tests/test_backend.cpp)

    #    target_compile_options(tgi_trtllm_backend_tests PRIVATE -Werror)
    target_link_directories(tgi_trtllm_backend_tests PRIVATE "${TRTLLM_NVRTC_WRAPPER_PARENT_LIBRARY_PATH}")
    target_include_directories(tgi_trtllm_backend_tests PUBLIC "${trtllm_SOURCE_DIR}/cpp/include")
    target_include_directories(tgi_trtllm_backend_tests PUBLIC "csrc/")
    target_link_libraries(tgi_trtllm_backend_tests PRIVATE ${TRTLLM_LIBS} CUDA::cudart CUDA::nvml)
    target_link_libraries(tgi_trtllm_backend_tests PUBLIC Catch2::Catch2WithMain nlohmann_json::nlohmann_json spdlog::spdlog tgi_trtllm_backend_impl)
    target_link_libraries(tgi_trtllm_backend_tests PRIVATE tensorrt_llm nvinfer_plugin_tensorrt_llm tensorrt_llm_nvrtc_wrapper)

    if (${TGI_TRTLLM_BACKEND_ENABLE_ASAN})
        message(STATUS "Enabled AddressSanitizer")
        target_link_options(tgi_trtllm_backend_tests BEFORE PUBLIC -fsanitize=address)
    endif ()

    if (${TGI_TRTLLM_BACKEND_ENABLE_UBSAN})
        message(STATUS "Enabled UndefinedSanitizer")
        target_link_options(tgi_trtllm_backend_tests BEFORE PUBLIC -fsanitize=undefined)
    endif ()

    install(TARGETS tgi_trtllm_backend_tests)

    #    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    #    include(CTest)
    #    include(Catch)
    #    catch_discover_tests(tgi_trtllm_backend_tests)
endif ()
