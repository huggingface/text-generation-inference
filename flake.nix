{
  inputs = {
    crate2nix = {
      url = "github:nix-community/crate2nix";
      inputs.nixpkgs.follows = "tgi-nix/nixpkgs";
    };
    nix-filter.url = "github:numtide/nix-filter";
    tgi-nix.url = "github:danieldk/tgi-nix";
    nixpkgs.follows = "tgi-nix/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "tgi-nix/nixpkgs";
    };
  };
  outputs =
    {
      self,
      crate2nix,
      nix-filter,
      nixpkgs,
      flake-utils,
      rust-overlay,
      tgi-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        cargoNix = crate2nix.tools.${system}.appliedCargoNix {
          name = "tgi";
          src = ./.;
          additionalCargoNixArgs = [ "--all-features" ];
        };
        pkgs = import nixpkgs {
          inherit system;
          inherit (tgi-nix.lib) config;
          overlays = [
            rust-overlay.overlays.default
            tgi-nix.overlays.default
          ];
        };
        crateOverrides = import ./nix/crate-overrides.nix { inherit pkgs nix-filter; };
        benchmark = cargoNix.workspaceMembers.text-generation-benchmark.build.override {
          inherit crateOverrides;
        };
        launcher = cargoNix.workspaceMembers.text-generation-launcher.build.override {
          inherit crateOverrides;
        };
        router = cargoNix.workspaceMembers.text-generation-router-v3.build.override {
          inherit crateOverrides;
        };
        server = pkgs.python3.pkgs.callPackage ./nix/server.nix { inherit nix-filter; };
      in
      {
        devShells = with pkgs; rec {
          default = pure;

          pure = mkShell {
            buildInputs = [
              benchmark
              launcher
              router
              server
            ];
          };

          impure = mkShell {
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
                protobuf
              ]
              ++ (with python3.pkgs; [
                venvShellHook
                pip
                ipdb
              ]);

            inputsFrom = [ server ];

            venvDir = "./.venv";

            postVenv = ''
              unset SOURCE_DATE_EPOCH
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
              export PATH=$PATH:~/.cargo/bin
            '';
          };
        };

        packages.default = pkgs.writeShellApplication {
          name = "text-generation-inference";
          runtimeInputs = [
            server
            router
          ];
          text = ''
            ${launcher}/bin/text-generation-launcher "$@"
          '';
        };
      }
    );
}
