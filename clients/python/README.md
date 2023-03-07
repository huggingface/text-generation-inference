# Text Generation

The Hugging Face Text Generation Python library provides a convenient way of interfacing with a
`text-generation-inference` instance running on your own infrastructure or on the Hugging Face Hub.

## Get Started

### Install

```shell
pip install text-generation
```

### Usage

```python
from text_generation import InferenceAPIClient

client = InferenceAPIClient("bigscience/bloomz")
text = client.generate("Why is the sky blue?").generated_text
print(text)
# ' Rayleigh scattering'

# Token Streaming
text = ""
for response in client.generate_stream("Why is the sky blue?"):
    if not response.token.special:
        text += response.token.text

print(text)
# ' Rayleigh scattering'
```

or with the asynchronous client:

```python
from text_generation import InferenceAPIAsyncClient

client = InferenceAPIAsyncClient("bigscience/bloomz")
response = await client.generate("Why is the sky blue?")
print(response.generated_text)
# ' Rayleigh scattering'

# Token Streaming
text = ""
async for response in client.generate_stream("Why is the sky blue?"):
    if not response.token.special:
        text += response.token.text

print(text)
# ' Rayleigh scattering'
```
