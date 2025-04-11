{
  nix-filter,
  buildPythonPackage,
  poetry-core,
  mypy-protobuf,
  awq-inference-engine,
  causal-conv1d,
  compressed-tensors,
  einops,
  exllamav2,
  flashinfer,
  flash-attn,
  flash-attn-layer-norm,
  flash-attn-v1,
  grpc-interceptor,
  grpcio-reflection,
  grpcio-status,
  grpcio-tools,
  hf-transfer,
  hf-xet,
  kernels,
  loguru,
  mamba-ssm,
  moe,
  opentelemetry-api,
  opentelemetry-exporter-otlp,
  opentelemetry-instrumentation-grpc,
  opentelemetry-semantic-conventions,
  outlines,
  paged-attention,
  peft,
  pillow,
  prometheus-client,
  punica-kernels,
  py-cpuinfo,
  pydantic,
  quantization,
  quantization-eetq,
  rotary,
  safetensors,
  tokenizers,
  torch,
  sentencepiece,
  transformers,
  typer,
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
    "pillow"
    "sentencepiece"
    "typer"
  ];

  pythonRemoveDeps = [ "scipy" ];

  dependencies = [
    awq-inference-engine
    causal-conv1d
    compressed-tensors
    einops
    exllamav2
    flashinfer
    flash-attn
    flash-attn-layer-norm
    grpc-interceptor
    grpcio-reflection
    grpcio-status
    grpcio-tools
    hf-transfer
    hf-xet
    kernels
    loguru
    mamba-ssm
    moe
    opentelemetry-api
    opentelemetry-exporter-otlp
    opentelemetry-instrumentation-grpc
    opentelemetry-semantic-conventions
    outlines
    paged-attention
    peft
    pillow
    prometheus-client
    punica-kernels
    py-cpuinfo
    pydantic
    quantization
    quantization-eetq
    rotary
    safetensors
    sentencepiece
    tokenizers
    transformers
    typer
  ];

  prePatch = ''
    python -m grpc_tools.protoc -Iproto/v3 --python_out=server/text_generation_server/pb \
           --grpc_python_out=server/text_generation_server/pb --mypy_out=server/text_generation_server/pb proto/v3/generate.proto
    find server/text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
    touch server/text_generation_server/pb/__init__.py
    cd server
  '';
}
