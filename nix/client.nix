{
  buildPythonPackage,
  poetry-core,
  huggingface-hub,
  pydantic,
}:

buildPythonPackage {
  name = "text-generation-x";

  src = ../clients/python;

  pyproject = true;

  build-system = [ poetry-core ];

  nativeBuildInputs = [ ];

  pythonRemoveDeps = [ ];

  dependencies = [
    huggingface-hub
    pydantic
  ];
}
