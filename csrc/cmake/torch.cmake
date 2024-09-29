fetchcontent_declare(
    torch
    URL https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip
#    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(torch)
list(APPEND CMAKE_PREFIX_PATH ${torch_SOURCE_DIR})
