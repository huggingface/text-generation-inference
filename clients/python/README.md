# Text Generation

The Hugging Face Text Generation Python library provides a convenient way of interfacing with a
`text-generation-inference` instance running on
[Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints) or on the Hugging Face Hub.

## Get Started

### Install

```shell
pip install text-generation
```

### Inference API Usage

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

### Hugging Face Inference Endpoint usage

```python
from text_generation import Client

endpoint_url = "https://YOUR_ENDPOINT.endpoints.huggingface.cloud"

client = Client(endpoint_url)
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
from text_generation import AsyncClient

endpoint_url = "https://YOUR_ENDPOINT.endpoints.huggingface.cloud"

client = AsyncClient(endpoint_url)
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

### Types

```python
# Prompt tokens
class PrefillToken:
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    # Optional since the logprob of the first token cannot be computed
    logprob: Optional[float]


# Generated tokens
class Token:
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: float
    # Is the token a special token
    # Can be used to ignore tokens when concatenating
    special: bool


# Generation finish reason
class FinishReason(Enum):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


# Additional sequences when using the `best_of` parameter
class BestOfSequence:
    # Generated text
    generated_text: str
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]
    # Prompt tokens
    prefill: List[PrefillToken]
    # Generated tokens
    tokens: List[Token]


# `generate` details
class Details:
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]
    # Prompt tokens
    prefill: List[PrefillToken]
    # Generated tokens
    tokens: List[Token]
    # Additional sequences when using the `best_of` parameter
    best_of_sequences: Optional[List[BestOfSequence]]


# `generate` return value
class Response:
    # Generated text
    generated_text: str
    # Generation details
    details: Details


# `generate_stream` details
class StreamDetails:
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]


# `generate_stream` return value
class StreamResponse:
    # Generated token
    token: Token
    # Complete generated text
    # Only available when the generation is finished
    generated_text: Optional[str]
    # Generation details
    # Only available when the generation is finished
    details: Optional[StreamDetails]
```