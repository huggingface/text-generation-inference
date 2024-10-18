cmake_minimum_required(VERSION 3.24)

project(tgi-llama-cpp-backend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

set(LLAMA_CPP_TARGET_VERSION "b3837" CACHE STRING "Version of llama.cpp to build against")
set(LLAMA_CPP_TARGET_CUDA_ARCHS "75-real;80-real;86-real;89-real;90-real" CACHE STRING "CUDA arch(s) to build")
option(LLAMA_CPP_BUILD_OFFLINE_RUNNER "Flag to build the standalone c++ backend runner")
option(LLAMA_CPP_BUILD_CUDA "Flag to build CUDA enabled inference through llama.cpp")

# Add dependencies
include(cmake/fmt.cmake)
include(cmake/spdlog.cmake)

if (${LLAMA_CPP_BUILD_CUDA})
    message(STATUS "Enabling llama.cpp CUDA support")

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${LLAMA_CPP_TARGET_CUDA_ARCHS})
    endif ()
    set(GGML_CUDA ON)
endif ()

# Download llama.cpp repo at the specific version
fetchcontent_declare(
        llama
        #    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG b3837
        GIT_SHALLOW FALSE
)

fetchcontent_makeavailable(llama)

add_library(tgi_llama_cpp_backend_impl STATIC csrc/backend.hpp csrc/backend.cpp)
target_compile_features(tgi_llama_cpp_backend_impl PRIVATE cxx_std_11)
target_link_libraries(tgi_llama_cpp_backend_impl PUBLIC fmt::fmt spdlog::spdlog llama common)

if (${LLAMA_CPP_BUILD_OFFLINE_RUNNER})
    message(STATUS "Building llama.cpp offline runner")
    add_executable(tgi_llama_cpp_offline_runner offline/main.cpp)
    target_link_libraries(tgi_llama_cpp_offline_runner tgi_llama_cpp_backend_impl)
endif ()


