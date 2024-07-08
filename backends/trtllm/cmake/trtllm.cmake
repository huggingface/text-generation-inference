set(USE_CXX11_ABI ON)
set(NVTX_DISABLE OFF)
set(BUILD_PYT OFF)
set(BUILD_PYBIND OFF)
set(BUILD_MICRO_BENCHMARKS OFF)
set(BUILD_BENCHMARKS OFF)
set(BUILD_TESTS OFF)
set(TRT_INCLUDE_DIR ${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR})
set(TRT_LIB_DIR ${TGI_TRTLLM_BACKEND_TRT_LIB_DIR})
set(CMAKE_CUDA_ARCHITECTURES ${TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST})

message(STATUS "Building for CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(FAST_BUILD ON)
else ()
    set(FAST_BUILD OFF)
endif ()

# This line turn off DEBUG in TRTLLM logger which is quite spammy
add_compile_definitions(NDEBUG OFF)

fetchcontent_declare(
        trtllm
        GIT_REPOSITORY https://github.com/nvidia/tensorrt-llm.git
        GIT_TAG 9dbc5b38baba399c5517685ecc5b66f57a177a4c
        GIT_SHALLOW TRUE
)
fetchcontent_makeavailable(trtllm)
message(STATUS "Found TensorRT-LLM: ${trtllm_SOURCE_DIR}")
execute_process(COMMAND git lfs install WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")
execute_process(COMMAND git lfs pull WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")
add_subdirectory("${trtllm_SOURCE_DIR}/cpp")
include_directories("${trtllm_SOURCE_DIR}/cpp/include")
