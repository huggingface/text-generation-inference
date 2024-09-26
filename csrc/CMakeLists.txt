cmake_minimum_required(VERSION 3.22)
project(text-generation-inference LANGUAGES C CXX CUDA)

# Update some policies
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()


# Define some overall constants
set(CMAKE_CXX_STANDARD 20)
set(TORCH_VERSION "2.3.1" "Version of PyTorch to build against")

# Define options
option(TGI_BUILD_CCL "Flag to enable/disable build of tgiccl collective library" ON)

# Add some modules
include(FetchContent)
include(cmake/spdlog.cmake)

# Let's find LibTorch
include(cmake/torch.cmake)
find_package(Python3 COMPONENTS Interpreter)
ProbeForPyTorchInstall()
ConfigurePyTorch()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include submodules
if (${TGI_BUILD_CCL})
    add_subdirectory(tgiccl)
endif ()
