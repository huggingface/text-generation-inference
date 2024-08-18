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
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "tgi-nix/nixpkgs";
    };
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
      poetry2nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        cargoNix = crate2nix.tools.${system}.appliedCargoNix {
          name = "tgi";
          src = ./.;
          additionalCargoNixArgs = [ "--all-features" ];
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
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEditablePackage;
        text-generation-server = mkPoetryEditablePackage { editablePackageSources = ./server; };
        crateOverrides = import ./nix/crate-overrides.nix { inherit pkgs nix-filter; };
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
      }
    );
}
