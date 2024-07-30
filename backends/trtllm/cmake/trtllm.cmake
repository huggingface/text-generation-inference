set(TRT_INCLUDE_DIR ${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR})
set(TRT_LIB_DIR ${TGI_TRTLLM_BACKEND_TRT_LIB_DIR})

set(USE_CXX11_ABI ON)
set(BUILD_PYT OFF)
set(BUILD_PYBIND OFF)
set(BUILD_MICRO_BENCHMARKS OFF)
set(BUILD_BENCHMARKS OFF)
set(BUILD_TESTS OFF)
set(CMAKE_CUDA_ARCHITECTURES ${TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST})

message(STATUS "Building for CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(FAST_BUILD ON)
    set(NVTX_DISABLE OFF)
else ()
    set(FAST_BUILD OFF)
    set(FAST_MATH ON)
    set(NVTX_DISABLE ON)
endif ()

fetchcontent_declare(
        trtllm
        GIT_REPOSITORY https://github.com/NVIDIA/TensorRT-LLM.git
        GIT_TAG a681853d3803ee5893307e812530b5e7004bb6e1
        GIT_SHALLOW FALSE
)
fetchcontent_makeavailable(trtllm)

message(STATUS "Found TensorRT-LLM: ${trtllm_SOURCE_DIR}")
execute_process(COMMAND git lfs install WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")
execute_process(COMMAND git lfs pull WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")

# TRTLLM use a JIT based *precompiled* library to generate some specific kernels, we are generating the path to this one here
set(TRTLLM_NVRTC_LIBRARY_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}tensorrt_llm_nvrtc_wrapper${CMAKE_SHARED_LIBRARY_SUFFIX}" CACHE INTERNAL "nvrtc wrapper library name")
set(TRTLLM_NVRTC_WRAPPER_LIBRARY_PATH "${trtllm_SOURCE_DIR}/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/${CMAKE_LIBRARY_ARCHITECTURE}/${TRTLLM_NVRTC_LIBRARY_NAME}"
        CACHE INTERNAL "nvrtc wrapper library path")

# The same Executor Static library
set(TRTLLM_EXECUTOR_STATIC_LIBRARY_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}tensorrt_llm_executor_static${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE INTERNAL "executor_static library name")
set(TRTLLM_EXECUTOR_STATIC_LIBRARY_PATH "${trtllm_SOURCE_DIR}/cpp/tensorrt_llm/executor/${CMAKE_LIBRARY_ARCHITECTURE}/${TRTLLM_EXECUTOR_STATIC_LIBRARY_NAME}" CACHE INTERNAL "executor_static library path")
