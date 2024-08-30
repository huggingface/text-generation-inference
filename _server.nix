{
  mkPoetryApplication,
  pkg-config,
  protobuf,
  openssl,
}:

mkPoetryApplication {
  # name = "text-generation-server";

  projectDir = ./server;

  # nativeBuildInputs = [ pkg-config ];

  # buildInputs = [ openssl.dev protobuf ];

}
