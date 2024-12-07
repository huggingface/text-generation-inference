## Compiling with MacOS

To compile the Llama.cpp backend on MacOS, you need to install `clang` and `cmake` via Homebrew:

```bash
brew install llvm cmake
```

You then need to configure CMakelists.txt to use the newly installed clang compiler.
You can do this by configuring your IDE or adding the following lines to the top of the file:

```cmake
set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)
```

CMakelist.txt assumes that Homebrew installs libc++ in `$HOMEBREW_PREFIX/opt/llvm/lib/c++`.