cmake_minimum_required(VERSION 3.20)

project(tgi-llama-cpp-backend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

set(LLAMA_CPP_TARGET_VERSION "b3837" STRING "Version of llama.cpp to build against")


# Add dependencies
include(cmake/fmt.cmake)
include(cmake/spdlog.cmake)

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
target_link_libraries(tgi_llama_cpp_backend_impl fmt::fmt spdlog::spdlog llama common)
