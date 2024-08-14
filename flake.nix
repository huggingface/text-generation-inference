{
  inputs = {
    crate2nix = {
      url = "github:nix-community/crate2nix";
      inputs.nixpkgs.follows = "tgi-nix/nixpkgs";
    };
    tgi-nix.url = "github:danieldk/tgi-nix";
    nixpkgs.follows = "tgi-nix/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    naersk = {
      url = "github:nix-community/naersk";
      inputs.nixpkgs.follows = "tgi-nix/nixpkgs";
    };
    poetry2nix.url = "github:nix-community/poetry2nix";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "tgi-nix/nixpkgs";
    };
  };
  outputs =
    {
      self,
      crate2nix,
      naersk,
      nixpkgs,
      flake-utils,
      rust-overlay,
      tgi-nix,
      poetry2nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        cargoNix = crate2nix.tools.${system}.appliedCargoNix {
          name = "tgi";
          src = ./.;
        };
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
        naersk' = pkgs.callPackage naersk { };
        router =
          with pkgs;
          naersk'.buildPackage {
            name = "router";
            src = ./.;
            cargoBuildOptions =
              x:
              x
              ++ [
                "-p"
                "text-generation-router-v3"
              ];
            nativeBuildInputs = [ pkg-config ];
            buildInputs = [
              openssl.dev
              protobuf
            ];
            doCheck = false;
          };

        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEditablePackage;
        text-generation-server = mkPoetryEditablePackage { editablePackageSources = ./server; };
      in
      {
        defaultPackage = router;
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

                causal-conv1d
                click
                einops
                exllamav2
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
                mamba-ssm
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

                cargoNix.workspaceMembers.text-generation-launcher.build
                router
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
