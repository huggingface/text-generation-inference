# ProbeForPyTorchInstall
# Attempts to find a Torch installation and set the Torch_ROOT variable
# based on introspecting the python environment. This allows a subsequent
# call to find_package(Torch) to work.
function(ProbeForPyTorchInstall)
    if (Torch_ROOT)
        message(STATUS "Using cached Torch root = ${Torch_ROOT}")
    else ()
        message(STATUS "Checking for PyTorch using ${Python3_EXECUTABLE} ...")
        execute_process(
                COMMAND ${Python3_EXECUTABLE}
                -c "import os;import torch;print(torch.utils.cmake_prefix_path, end='')"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE PYTORCH_STATUS
                OUTPUT_VARIABLE PYTORCH_PACKAGE_DIR)
        if (NOT PYTORCH_STATUS EQUAL "0")
            message(STATUS "Unable to 'import torch' with ${Python3_EXECUTABLE} (fallback to explicit config)")
            return()
        endif ()
        message(STATUS "Found PyTorch installation at ${PYTORCH_PACKAGE_DIR}")

        set(Torch_ROOT "${PYTORCH_PACKAGE_DIR}" CACHE STRING
                "Torch configure directory" FORCE)
    endif ()
endfunction()


# ConfigurePyTorch
# Extensions compiled against PyTorch must be ABI-compatible with PyTorch.
# On Linux, there are two components to this:
#   1) Dual ABI settings for libstdc++
#      See https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
# For this, PyTorch helpfully provides a function to check which ABI it was
# compiled against.
#   2) C++ ABI compatibility version
#      See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html (Sec 5/6)
# The second is a bit more complicated. GCC has official compatibility strings
# which can be specified by -fabi-version. Clang has no notion of ABI
# versioning (https://lists.llvm.org/pipermail/cfe-dev/2015-June/043735.html).
# Separately, pybind11 keeps an internal variable which records its ABI info
# (PYBIND11_INTERNALS_ID in include/pybind11/detail/internals.h). Differences
# in this variable between torch-mlir and PyTorch will cause type errors.
# Thus, our best option is to:
#   a) Identify which ABI version PyTorch was compiled with
#   b) Tell gcc to use that version
#     or
#   c) Tell clang to pretend to use it and hope it's ABI-compatible, and
#      tell pybind to pretend we're gcc.
#
# MacOS does not have a dual ABI problem.
# FIXME: I don't know if MacOS needs ABI compatibility version flags.
#
# In the future, we may want to switch away from custom building these
# extensions and instead rely on the Torch machinery directly (definitely want
# to do that for official builds).
function(ConfigurePyTorch)
    message(STATUS "Checking PyTorch ABI settings...")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        # Check dual ABI setting first
        execute_process(
                COMMAND ${Python3_EXECUTABLE}
                -c "import torch; import sys; sys.stdout.write('1' if torch.compiled_with_cxx11_abi() else '0')"
                RESULT_VARIABLE _result
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE _use_cxx11_abi)
        if (_result)
            message(FATAL_ERROR "Failed to determine C++ Dual ABI: ${Python3_EXECUTABLE} -> ${_result}")
        endif ()
        message(STATUS "PyTorch C++ Dual ABI setting: \"${_use_cxx11_abi}\"")

        # Check ABI compatibility version
        execute_process(
                COMMAND ${Python3_EXECUTABLE}
                -c "import torch; import sys; abi=torch._C._PYBIND11_BUILD_ABI; abi.startswith('_cxxabi10') or sys.exit(1); sys.stdout.write(str(abi[-2:]))"
                RESULT_VARIABLE _result
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE _cxx_abi_version)
        if (_result)
            message(FATAL_ERROR "Failed to determine C++ ABI version")
        endif ()
        message(STATUS "PyTorch C++ ABI version: \"${_cxx_abi_version}\"")

        # Specialize compile flags for compiler
        if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
            set(TORCH_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=${_use_cxx11_abi} -fabi-version=${_cxx_abi_version}")
        elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
            set(TORCH_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=${_use_cxx11_abi} -U__GXX_ABI_VERSION -D__GXX_ABI_VERSION=10${_cxx_abi_version} '-DPYBIND11_COMPILER_TYPE=\"_gcc\"'")
        else ()
            message(WARNING "Unrecognized compiler. Cannot determine ABI flags.")
            return()
        endif ()
        set(TORCH_CXXFLAGS "${TORCH_CXXFLAGS}" PARENT_SCOPE)
    endif ()
endfunction()

function(ConfigureLibTorch)
    message(STATUS "Checking LibTorch ABI settings...")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        message(STATUS "libtorch_python is ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")
        # Check dual ABI setting first
        execute_process(
                COMMAND bash "-c" "cat ${TORCH_INSTALL_PREFIX}/share/cmake/Torch/TorchConfig.cmake | egrep -o '_GLIBCXX_USE_CXX11_ABI=[0-1]' | egrep -o '.$'"
                RESULT_VARIABLE _result
                OUTPUT_VARIABLE _use_cxx11_abi
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (_result)
            message(FATAL_ERROR "Failed to determine LibTorch C++ Dual ABI")
        endif ()
        message(STATUS "LibTorch C++ Dual ABI setting: \"${_use_cxx11_abi}\"")

        # Check ABI compatibility version
        execute_process(
                COMMAND bash "-c" "strings ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so | egrep '^_cxxabi[0-9]{4}' | egrep -o '..$'"
                RESULT_VARIABLE _result
                OUTPUT_VARIABLE _cxx_abi_version
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (_result)
            message(FATAL_ERROR "Failed to determine LibTorch C++ ABI version")
        endif ()
        message(STATUS "LibTorch C++ ABI version: \"${_cxx_abi_version}\"")

        # Specialize compile flags for compiler
        if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
            set(TORCH_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=${_use_cxx11_abi} -fabi-version=${_cxx_abi_version}")
        elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
            set(TORCH_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=${_use_cxx11_abi} -U__GXX_ABI_VERSION -D__GXX_ABI_VERSION=10${_cxx_abi_version} '-DPYBIND11_COMPILER_TYPE=\"_gcc\"'")
        else ()
            message(WARNING "Unrecognized compiler. Cannot determine ABI flags.")
            return()
        endif ()
        set(TORCH_CXXFLAGS "${TORCH_CXXFLAGS}" PARENT_SCOPE)
    endif ()
endfunction()

function(torch_mlir_python_target_compile_options target)
    target_compile_options(${target} PRIVATE
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            # Enable RTTI and exceptions.
            -frtti -fexceptions
            # Noisy pybind warnings
            -Wno-unused-value
            -Wno-covered-switch-default
            >
            $<$<CXX_COMPILER_ID:MSVC>:
            # Enable RTTI and exceptions.
            /EHsc /GR>
    )
endfunction()