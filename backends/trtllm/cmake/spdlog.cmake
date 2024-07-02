set(SPDLOG_USE_FMT ON)
fetchcontent_declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v2.x
)
fetchcontent_makeavailable(spdlog)