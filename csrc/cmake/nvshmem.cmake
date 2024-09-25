if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(NVSHMEM_DEBUG OFF)
    set(NVSHMEM_VERBOSE OFF)
else ()
    set(NVSHMEM_DEBUG ON)
    set(NVSHMEM_VERBOSE ON)
endif ()

fetchcontent_declare(
        nvshmem
        URL https://developer.download.nvidia.com/compute/redist/nvshmem/3.0.6/source/nvshmem_src_3.0.6-4.txz
        DOWNLOAD_EXTRACT_TIMESTAMP
)

fetchcontent_makeavailable(nvshmem)