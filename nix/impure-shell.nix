{
  lib,
  mkShell,
  black,
  cmake,
  isort,
  ninja,
  which,
  cudaPackages,
  openssl,
  ffmpeg,
  llvmPackages,
  gcc,
  gcc-unwrapped,
  stdenv,
  pkg-config,
  poetry,
  protobuf,
  python3,
  pyright,
  redocly,
  ruff,
  rust-bin,
  server,

  # Enable dependencies for building CUDA packages. Useful for e.g.
  # developing marlin/moe-kernels in-place.
  withCuda ? false,
}:

mkShell {
  nativeBuildInputs =
    [
      black
      isort
      pkg-config
      poetry
      (rust-bin.stable.latest.default.override {
        extensions = [
          "rust-analyzer"
          "rust-src"
        ];
      })
      protobuf
      pyright
      redocly
      ruff
    ]
    ++ (lib.optionals withCuda [
      cmake
      ninja
      which

      # For most Torch-based extensions, setting CUDA_HOME is enough, but
      # some custom CMake builds (e.g. vLLM) also need to have nvcc in PATH.
      cudaPackages.cuda_nvcc
    ]);
  buildInputs =
    [
      openssl.dev
      ffmpeg.dev
      llvmPackages.libclang
      gcc.cc
      gcc-unwrapped
      stdenv
    ]
    ++ (with python3.pkgs; [
      venvShellHook
      docker
      pip
      ipdb
      click
      pytest
      pytest-asyncio
      syrupy
    ])
    ++ (lib.optionals withCuda (
      with cudaPackages;
      [
        cuda_cccl
        cuda_cudart
        cuda_nvrtc
        cuda_nvtx
        cuda_profiler_api
        cudnn
        libcublas
        libcusolver
        libcusparse
      ]
    ));

  inputsFrom = [ server ];

  env = {
    LIBCLANG_PATH = "${llvmPackages.libclang.lib}/lib";
    CPATH = "${gcc-unwrapped}/lib/gcc/${stdenv.hostPlatform.config}/${gcc-unwrapped.version}/include";
    BINDGEN_EXTRA_CLANG_ARGS = builtins.concatStringsSep " " [
      "-I${gcc.libc.dev}/include"
      "-I${gcc}/lib/gcc/x86_64-unknown-linux-gnu/${gcc.version}/include"
      "-I${llvmPackages.libclang.lib}/lib/clang/${llvmPackages.libclang.version}/include"
    ];
  } // lib.optionalAttrs withCuda {
    CUDA_HOME = "${lib.getDev cudaPackages.cuda_nvcc}";
    TORCH_CUDA_ARCH_LIST = lib.concatStringsSep ";" python3.pkgs.torch.cudaCapabilities;
  };

  venvDir = "./.venv";

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    ( cd server ; python -m pip install --no-dependencies -e . )
    ( cd clients/python ; python -m pip install --no-dependencies -e . )
  '';

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export PATH=$PATH:~/.cargo/bin
  '';
}
