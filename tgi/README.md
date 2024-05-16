# TGI (Python Package)

> [!IMPORTANT]
> This is an experimental package and intended for research purposes only. The package is likely to change and should only be used for testing and development.

`tgi` is a simple Python package that wraps the `text-generation-server` and `text-generation-launcher` packages. It provides a simple interface to the text generation server.

```bash
make install
# this compiles the code and runs pip install for `tgi`
```

## Usage

See the full example in the [`app.py`](./app.py) file.

```python
from tgi import TGI
from huggingface_hub import InferenceClient
import time

llm = TGI(model_id="google/paligemma-3b-mix-224")

# ✂️ startup logic snipped
print("Model is ready!")

client = InferenceClient("http://localhost:3000")
generated = client.text_generation("What are the main characteristics of a cat?")
print(generated)

# Cats are known for their independent nature, curious minds, and affectionate nature. Here are the main characteristics of a cat...

llm.close()
```

## How it works

Technically this is a `pyo3` package that wraps the `text-generation-server` and `text-generation-launcher` packages, and slightly modifies the launcher to rely on the interal code rather than launch an external binary.

## Known issues/limitations

- [ ] server does not gracefully handle shutdowns (trying to avoid python context for better notebook dev experience)
- [ ] issues with tracing (launcher and router should share tracer)
- [ ] text-generation-server is not integrated and still relies on the external install
- [ ] not all parameters are exposed/passed through
- [ ] general cleanup and refactoring needed
- [ ] review naming and explore pushing to PyPi
