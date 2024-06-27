cmake_minimum_required(VERSION 3.20)

project(tgi_trtllm_backend)
set(CMAKE_CXX_STANDARD 20)

include(ExternalProject)

# Projects user-defined variables
set(TGI_TRTLLM_BACKEND_TRTLLM_ROOT_DIR CACHE PATH "Root path where to find TensorRT-LLM")

# Project variables
set(TGI_TRTLLM_BACKEND_SOURCES backend.cpp)

if (NOT EXISTS ${TGI_TRTLLM_BACKEND_TRTLLM_ROOT_DIR})
    message(FATAL_ERROR "Cannot find TensorRT-LLM at ${TGI_TRTLLM_BACKEND_TRTLLM_ROOT_DIR}")
endif ()

message(STATUS "Found TensorRT-LLM ${TGI_TRTLLM_BACKEND_TRTLLM_ROOT_DIR}")

# Set some TensorRT-LLM specific variables
set(TRTLLM_ROOT_DIR ${TGI_TRTLLM_BACKEND_TRTLLM_ROOT_DIR})
set(TRTLLM_INCLUDE_DIR "${TRTLLM_ROOT_DIR}/include")
set(TRTLLM_SOURCES_DIR "${TRTLLM_ROOT_DIR}/cpp")

# Ensure CUDA is found
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

#include(${TRTLLM_ROOT_DIR}/cpp/cmake/modules/set_ifndef.cmake)
#include(${TRTLLM_ROOT_DIR}/cpp/cmake/modules/find_library_create_target.cmake)
#
#set(CMAKE_CXX_FLAGS "-Wall -pthread -lstdc++")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#
## TRT dependencies
#set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
#set_ifndef(TRT_INCLUDE_DIR /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
#set(TRT_LIB nvinfer)
## On Windows major version is appended to nvinfer libs.
#if (WIN32)
#    set(TRT_LIB_NAME nvinfer_10)
#else ()
#    set(TRT_LIB_NAME nvinfer)
#endif ()
#find_library_create_target(${TRT_LIB} ${TRT_LIB_NAME} SHARED ${TRT_LIB_DIR})
#message(${TRT_INCLUDE_DIR})
#include_directories("${TRT_INCLUDE_DIR}")
#
#if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
#    add_definitions("-DENABLE_BF16")
#    message(
#            STATUS
#            "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
#    )
#endif ()
#
#if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
#    add_definitions("-DENABLE_FP8")
#    message(
#            STATUS
#            "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
#    )
#endif ()
#
## tensorrt_llm shared lib
#add_library(tensorrt_llm SHARED IMPORTED)
#set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_LIB_PATH})
#set_property(
#        TARGET tensorrt_llm PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES
#        CUDA::cuda_driver CUDA::cudart_static CUDA::nvml)
#
## nvinfer_plugin_tensorrt_llm shared lib
#add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
#set_property(TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_PLUGIN_PATH})
#set_property(TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES tensorrt_llm)
#
#include_directories(${TRTLLM_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

add_library(tgi_trtllm_backend ${TGI_TRTLLM_BACKEND_SOURCES})
target_link_libraries(tgi_trtllm_backend)