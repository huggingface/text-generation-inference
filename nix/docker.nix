{
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
  config = {
    EntryPoint = [ "${text-generation-inference}/bin/text-generation-inference" ];
    Env = [
      "HF_HOME=/data"
      "PORT=80"
    ];

  };
  contents = [ cacert ];
}
