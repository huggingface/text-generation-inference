{
  buildPythonPackage,
  poetry-core,
  aiohttp,
  huggingface-hub,
  pydantic,
}:

buildPythonPackage {
  name = "text-generation";

  src = ../clients/python;

  pyproject = true;

  build-system = [ poetry-core ];

  dependencies = [
    aiohttp
    huggingface-hub
    pydantic
  ];
}
