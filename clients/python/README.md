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

Check all currently deployed models on the Huggingface Inference API with `Text Generation` support:

```python
from text_generation.inference_api import deployed_models

print(deployed_models())
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
# Request Parameters
class Parameters:
    # Activate logits sampling
    do_sample: bool
    # Maximum number of generated tokens
    max_new_tokens: int
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float]
    # Whether to prepend the prompt to the generated text
    return_full_text: bool
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str]
    # Random sampling seed
    seed: Optional[int]
    # The value used to module the logits distribution.
    temperature: Optional[float]
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int]
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float]
    # truncate inputs tokens to the given size
    truncate: Optional[int]
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float]
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int]
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool
    # Get decoder input token logprobs and ids
    decoder_input_details: bool

# Decoder input tokens
class InputToken:
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
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
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
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
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

# Inference API currently deployed model
class DeployedModel:
    model_id: str
    sha: str
```