{
  buildPythonPackage,
  poetry-core,
  huggingface-hub,
  pydantic,
}:

buildPythonPackage {
  name = "text-generation";

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
