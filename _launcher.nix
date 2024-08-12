{ buildRustPackage, importCargoLock, pkg-config, protobuf, openssl }:

buildRustPackage {
  name = "text-generation-lancher";

  src = ./.;

  sourceDir = ./launcher;

  cargoLock = {
    lockFile = ./Cargo.lock;
  };

  nativeBuildInputs = [ pkg-config ];

  buildInputs = [ openssl.dev protobuf ];

}
