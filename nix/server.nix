{
  nix-filter,
  buildPythonPackage,
  poetry-core,
  mypy-protobuf,
  awq-inference-engine,
  causal-conv1d,
  eetq,
  einops,
  exllamav2,
  fbgemm-gpu,
  flashinfer,
  flash-attn,
  flash-attn-layer-norm,
  flash-attn-rotary,
  grpc-interceptor,
  grpcio-reflection,
  grpcio-status,
  grpcio-tools,
  hf-transfer,
  loguru,
  mamba-ssm,
  marlin-kernels,
  moe-kernels,
  opentelemetry-api,
  opentelemetry-exporter-otlp,
  opentelemetry-instrumentation-grpc,
  opentelemetry-semantic-conventions,
  peft,
  punica-kernels,
  safetensors,
  tokenizers,
  torch,
  sentencepiece,
  transformers,
  typer,
  vllm,
}:

let
  filter = nix-filter.lib;
in
buildPythonPackage {
  name = "text-generation-server";

  src = filter {
    root = ../.;
    include = with filter; [
      isDirectory
      (and (inDirectory "server") (or_ (matchExt "py") (matchExt "pyi")))
      "server/pyproject.toml"
      (and (inDirectory "proto/v3") (matchExt "proto"))
    ];
  };

  pyproject = true;

  build-system = [ poetry-core ];

  nativeBuildInputs = [ mypy-protobuf ];

  pythonRelaxDeps = [
    "einops"
    "huggingface-hub"
    "loguru"
    "opentelemetry-instrumentation-grpc"
    "sentencepiece"
    "typer"
  ];

  pythonRemoveDeps = [ "scipy" ];

  dependencies = [
    awq-inference-engine
    eetq
    causal-conv1d
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
    moe-kernels
    opentelemetry-api
    opentelemetry-exporter-otlp
    opentelemetry-instrumentation-grpc
    opentelemetry-semantic-conventions
    peft
    punica-kernels
    safetensors
    sentencepiece
    tokenizers
    transformers
    typer
    vllm
  ];

  prePatch = ''
    python -m grpc_tools.protoc -Iproto/v3 --python_out=server/text_generation_server/pb \
           --grpc_python_out=server/text_generation_server/pb --mypy_out=server/text_generation_server/pb proto/v3/generate.proto
    find server/text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
    touch server/text_generation_server/pb/__init__.py
    cd server
  '';
}
