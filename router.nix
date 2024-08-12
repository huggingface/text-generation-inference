{ buildRustPackage, importCargoLock, pkg-config, protobuf, openssl }:

buildRustPackage {
  name = "text-generation-router";

  src = ./.;

  sourceDir = ./backends/v3;

  cargoLock = {
    lockFile = ./Cargo.lock;
  };

  nativeBuildInputs = [ pkg-config ];

  buildInputs = [ openssl.dev protobuf ];

}
