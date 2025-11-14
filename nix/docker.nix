{
  stdenv,
  dockerTools,
  cacert,
  text-generation-inference,
  stream ? false,
}:

let
  build = if stream then dockerTools.streamLayeredImage else dockerTools.buildLayeredImage;
in
build {
  name = "tgi-docker";
  tag = "latest";
  compressor = "zstd";
  config = {
    EntryPoint = [ "${text-generation-inference}/bin/text-generation-inference" ];
    Env = [
      "HF_HOME=/data"
      "PORT=80"
      # The CUDA container toolkit will mount the driver shim into the
      # container. We just have to ensure that the dynamic loader finds
      # the libraries.
      "LD_LIBRARY_PATH=/usr/lib64"
    ];

  };
  extraCommands = ''
    mkdir -p tmp
    chmod -R 1777 tmp
  '';
  contents = [
    cacert
    stdenv.cc
  ];
}
