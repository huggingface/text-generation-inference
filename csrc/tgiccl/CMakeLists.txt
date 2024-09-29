project(tgiccl LANGUAGES C CXX CUDA)

set(TGICCL_HEADERS tgiccl.hpp TgiCclBackend.hpp)
set(TGICCL_SOURCES TgiCclBackend.cpp)

add_library(tgiccl STATIC ${TGICCL_HEADERS} ${TGICCL_SOURCES})
target_link_libraries(tgiccl PUBLIC tgi_common fmt::fmt spdlog::spdlog CUDA::nvml ${TORCH_LIBRARIES})

add_executable(test_tgiccl test_tgiccl.cpp)
target_link_libraries(test_tgiccl PUBLIC tgiccl fmt::fmt spdlog::spdlog ${TORCH_LIBRARIES})