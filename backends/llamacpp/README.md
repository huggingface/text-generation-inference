# Llamacpp backend

If all your dependencies are installed at the system level, running
cargo build should be sufficient. However, if you want to experiment
with different versions of llama.cpp, some additional setup is required.

## Install llama.cpp

    LLAMACPP_PREFIX=$(pwd)/llama.cpp.out

    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build \
        -DCMAKE_INSTALL_PREFIX="$LLAMACPP_PREFIX" \
        -DLLAMA_BUILD_COMMON=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF
    cmake --build build --config Release -j
    cmake --install build

## Build TGI

    PKG_CONFIG_PATH="$LLAMACPP_PREFIX/lib/pkgconfig" cargo build
