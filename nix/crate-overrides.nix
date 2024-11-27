{ pkgs, nix-filter }:

let
  filter = nix-filter.lib;
in
with pkgs;
defaultCrateOverrides
// {
  aws-lc-rs = attrs: {
    # aws-lc-rs does its own custom parsing of Cargo environment
    # variables like DEP_.*_INCLUDE. However buildRustCrate does
    # not use the version number, so the parsing fails.
    postPatch = ''
      substituteInPlace build.rs \
        --replace-fail \
        "assert!(!selected.is_empty()" \
        "// assert!(!selected.is_empty()"
    '';
  };
  rav1e = attrs: { env.CARGO_ENCODED_RUSTFLAGS = "-C target-feature=-crt-static"; };

  ffmpeg-sys-next = attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or []) ++ [
      pkg-config
    ];
    buildInputs = (attrs.buildInputs or []) ++ [
      llvmPackages.libclang
      ffmpeg.dev
      gcc.cc
      gcc-unwrapped
      stdenv
    ];
    env = (attrs.env or {}) // {
      LIBCLANG_PATH = "${llvmPackages.libclang.lib}/lib";
      CPATH = "${gcc-unwrapped}/lib/gcc/${stdenv.hostPlatform.config}/${gcc-unwrapped.version}/include";
      BINDGEN_EXTRA_CLANG_ARGS = builtins.concatStringsSep " " [
        "-I${gcc.libc.dev}/include"
        "-I${gcc}/lib/gcc/x86_64-unknown-linux-gnu/${gcc.version}/include"
        "-I${llvmPackages.libclang.lib}/lib/clang/${llvmPackages.libclang.version}/include"
      ];
      PKG_CONFIG_PATH = "${ffmpeg.dev}/lib/pkgconfig";
    };
  };

  grpc-metadata = attrs: {
    src = filter {
      root = ../backends/grpc-metadata;
      include = with filter; [
        isDirectory
        (matchExt "rs")
      ];
    };
  };
  pyo3-build-config = attrs: {
    buildInputs = [ python3 ];
  };
  text-generation-benchmark = attrs: {
    src = filter {
      root = ../benchmark;
      include = with filter; [
        isDirectory
        (matchExt "rs")
      ];
    };
  };
  text-generation-client = attrs: {
    src = filter {
      root = ../.;
      include = with filter; [
        isDirectory
        (and (inDirectory "backends/client") (matchExt "rs"))
        (and (inDirectory "proto") (matchExt "proto"))
      ];
    };
    postPatch = "cd backends/client";
    buildInputs = [ protobuf ];
  };
  text-generation-launcher = attrs: {
    src = filter {
      root = ../launcher;
      include = with filter; [
        isDirectory
        (matchExt "rs")
      ];
    };
  };
  text-generation-router = attrs: {
    src = filter {
      root = ../router;
      include = with filter; [
        isDirectory
        (matchExt "rs")
      ];
    };
  };
  text-generation-router-v3 = attrs: {
    # We need to do the src/source root dance so that the build
    # has access to the protobuf file.
    src = filter {
      root = ../.;
      include = with filter; [
        isDirectory
        (and (inDirectory "backends/v3") (matchExt "rs"))
        (and (inDirectory "proto") (matchExt "proto"))
      ];
    };
    postPatch = "cd backends/v3";
    buildInputs = [ protobuf ];
  };
}
