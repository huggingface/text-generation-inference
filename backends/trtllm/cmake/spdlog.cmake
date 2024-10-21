set(SPDLOG_USE_FMT ON)
set(SPDLOG_BUILD_SHARED OFF)
set(SPDLOG_FMT_EXTERNAL ON)

# Define the level at which SPDLOG_ compilation level is defined
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG)
else ()
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO)
endif ()

fetchcontent_declare(
        spdlog
        DOWNLOAD_EXTRACT_TIMESTAMP
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz
)
fetchcontent_makeavailable(spdlog)
