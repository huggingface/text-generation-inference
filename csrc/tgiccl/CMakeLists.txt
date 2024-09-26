project(tgiccl LANGUAGES C CXX CUDA)

set(TGICCL_HEADERS tgiccl.hpp TgiCclBackend.hpp)
set(TGICCL_SOURCES TgiCclBackend.cpp)

find_package(CUDAToolkit REQUIRED)

add_library(tgiccl SHARED ${TGICCL_HEADERS} ${TGICCL_SOURCES})
target_link_libraries(tgiccl PUBLIC spdlog::spdlog CUDA::nvml ${TORCH_LIBRARIES})

add_executable(test_tgiccl test_tgiccl.cpp)
target_link_libraries(test_tgiccl tgiccl spdlog::spdlog)