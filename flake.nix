{
  inputs = {
    tgi-nix.url = "github:danieldk/tgi-nix";
    nixpkgs.follows = "tgi-nix/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      tgi-nix,
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
          overlays = [ tgi-nix.overlay ];
        };
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            buildInputs =
              [
                cargo
                clippy
                openssl.dev
                pkg-config
                rustfmt
              ]
              ++ (with python3.pkgs; [
                venvShellHook
                pip

                einops
                fbgemm-gpu
                flash-attn
                flash-attn-layer-norm
                flash-attn-rotary
                grpc-interceptor
                grpcio-reflection
                grpcio-status
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
