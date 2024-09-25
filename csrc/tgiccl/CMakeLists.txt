project(tgiccl)

set(TGICCL_HEADER_FILES tgiccl.hpp)
#set(TGICCL_SOURCE_FILES)

add_library(tgiccl SHARED ${TGICCL_HEADER_FILES})
target_link_libraries(tgiccl nvshmem)