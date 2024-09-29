
set(TGI_COMMON_HEADERS include/common/device.hpp)
set(TGI_COMMON_SOURCES lib/device.cpp)

add_library(tgi_common STATIC ${TGI_COMMON_HEADERS} ${TGI_COMMON_SOURCES})
target_link_libraries(tgi_common fmt::fmt spdlog::spdlog ${TORCH_LIBRARIES} CUDA::nvml)

target_include_directories(tgi_common PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(tgi_common PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/>
)