{
  inputs = {
    tgi-nix.url = "github:danieldk/tgi-nix";
    nixpkgs.follows = "tgi-nix/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "tgi-nix/nixpkgs";
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      tgi-nix,
      poetry2nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
        pkgs = import nixpkgs {
          inherit config system;
          overlays = [
            rust-overlay.overlays.default
            tgi-nix.overlay
          ];
        };

        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEditablePackage;
        text-generation-server = mkPoetryEditablePackage {
            editablePackageSources = ./server;
        };
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            buildInputs =
              [
                openssl.dev
                pkg-config
                (rust-bin.stable.latest.default.override {
                  extensions = [
                    "rust-analyzer"
                    "rust-src"
                  ];
                })
              ]
              ++ (with python3.pkgs; [
                venvShellHook
                pip

                click
                einops
                fbgemm-gpu
                flashinfer
                flash-attn
                flash-attn-layer-norm
                flash-attn-rotary
                grpc-interceptor
                grpcio-reflection
                grpcio-status
                grpcio-tools
                hf-transfer
                loguru
                marlin-kernels
                opentelemetry-api
                opentelemetry-exporter-otlp
                opentelemetry-instrumentation-grpc
                opentelemetry-semantic-conventions
                peft
                tokenizers
                torch
                transformers
                vllm

                (callPackage ./router.nix {
                  inherit (rustPlatform) buildRustPackage importCargoLock;
                })
                (callPackage ./_launcher.nix {
                  inherit (rustPlatform) buildRustPackage importCargoLock;
                })
              ]);

            venvDir = "./.venv";

            postVenv = ''
              unset SOURCE_DATE_EPOCH
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
            '';
          };
      }
    );
}
