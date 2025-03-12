{
  stdenv,
  dockerTools,
  cacert,
  text-generation-inference,
  runCommand,
  stream ? false,
}:

let
  build = if stream then dockerTools.streamLayeredImage else dockerTools.buildLayeredImage;
  tmp = runCommand "tmp" { } ''
    mkdir $out
    mkdir -m 1777 $out/tmp
  '';
in
build {
  name = "tgi-docker";
  tag = "latest";
  config = {
    EntryPoint = [ "${text-generation-inference}/bin/text-generation-inference" ];
    Env = [
      "HF_HOME=/data"
      "PORT=80"
    ];

  };
  contents = [
    cacert
    stdenv.cc
    tmp
  ];
}
