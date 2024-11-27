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
    nativeBuildInputs = [
      pkg-config
    ];
    buildInputs = [
      rustPlatform.bindgenHook
      ffmpeg
    ];
  };

  ffmpeg-next = attrs: {
    # Somehow the variables that are passed are mangled, so they are not
    # correctly passed to the ffmpeg-next build script. Worth investigating
    # more since it's probably a bug in crate2nix or buildRustCrate.
    postPatch = ''
      substituteInPlace build.rs \
        --replace-fail "DEP_FFMPEG_" "DEP_FFMPEG_SYS_NEXT_"
    '';
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
