set(SPDLOG_USE_FMT ON)
set(SPDLOG_BUILD_SHARED OFF)
set(SPDLOG_FMT_EXTERNAL OFF)
set(SPDLOG_INSTALL ON)
set(SPDLOG_NO_ATOMIC_LEVELS ON)  # We are not modifying log levels concurrently

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(SPDLOG_CLOCK_COARSE ON)
endif ()

# Define the level at which SPDLOG_ compilation level is defined
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Verbose logging is enabled in debug build")
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
else ()
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
endif ()

fetchcontent_declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz
)
fetchcontent_makeavailable(spdlog)
