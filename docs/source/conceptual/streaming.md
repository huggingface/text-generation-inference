# Streaming

## What is Streaming?

Token streaming is the mode in which the server returns the tokens one by one as the model generates them. This enables showing progressive generations to the user rather than waiting for the whole generation. Streaming is an essential aspect of the end-user experience as it reduces latency, one of the most critical aspects of a smooth experience.

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/streaming-generation-visual_360.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/streaming-generation-visual-dark_360.gif"
    />
</div>

With token streaming, the server can start returning the tokens one by one before having to generate the whole response. Users can have a sense of the generation's quality before the end of the generation. This has different positive effects:

* Users can get results orders of magnitude earlier for extremely long queries.
* Seeing something in progress allows users to stop the generation if it's not going in the direction they expect.
* Perceived latency is lower when results are shown in the early stages.
* When used in conversational UIs, the experience feels more natural.

For example, a system can generate 100 tokens per second. If the system generates 1000 tokens, with the non-streaming setup, users need to wait 10 seconds to get results. On the other hand, with the streaming setup, users get initial results immediately, and although end-to-end latency will be the same, they can see half of the generation after five seconds. Below you can see an interactive demo that shows non-streaming vs streaming side-by-side. Click **generate** below.

<div class="block dark:hidden">
	<iframe
        src="https://osanseviero-streaming-vs-non-streaming.hf.space?__theme=light"
        width="850"
        height="350"
    ></iframe>
</div>
<div class="hidden dark:block">
    <iframe
        src="https://osanseviero-streaming-vs-non-streaming.hf.space?__theme=dark"
        width="850"
        height="350"
    ></iframe>
</div>

## How to use Streaming?

### Streaming with Python

To stream tokens with `InferenceClient`, simply pass `stream=True` and iterate over the response.

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://127.0.0.1:8080")
for token in client.text_generation("How do you make cheese?", max_new_tokens=12, stream=True):
    print(token)

# To
# make
# cheese
#,
# you
# need
# to
# start
# with
# milk
#.
```

If you want additional details, you can add `details=True`. In this case, you get a `TextGenerationStreamResponse` which contains additional information such as the probabilities and the tokens. For the final response in the stream, it also returns the full generated text.

```python
for details in client.text_generation("How do you make cheese?", max_new_tokens=12, details=True, stream=True):
    print(details)

#TextGenerationStreamResponse(token=Token(id=193, text='\n', logprob=-0.007358551, special=False), generated_text=None, details=None)
#TextGenerationStreamResponse(token=Token(id=2044, text='To', logprob=-1.1357422, special=False), generated_text=None, details=None)
#TextGenerationStreamResponse(token=Token(id=717, text=' make', logprob=-0.009841919, special=False), generated_text=None, details=None)
#...
#TextGenerationStreamResponse(token=Token(id=25, text='.', logprob=-1.3408203, special=False), generated_text='\nTo make cheese, you need to start with milk.', details=StreamDetails(finish_reason=<FinishReason.Length: 'length'>, generated_tokens=12, seed=None))
```

The `huggingface_hub` library also comes with an `AsyncInferenceClient` in case you need to handle the requests concurrently.

```python
from huggingface_hub import AsyncInferenceClient

client = AsyncInferenceClient("http://127.0.0.1:8080")
async for token in await client.text_generation("How do you make cheese?", stream=True):
    print(token)

# To
# make
# cheese
#,
# you
# need
# to
# start
# with
# milk
#.
```

### Streaming with cURL

To use the `generate_stream` endpoint with curl, you can add the `-N` flag, which disables curl default buffering and shows data as it arrives from the server

```curl
curl -N 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

### Streaming with JavaScript

First, we need to install the `@huggingface/inference` library.
`npm install @huggingface/inference`

If you're using the free Inference API, you can use `HfInference`. If you're using inference endpoints, you can use `HfInferenceEndpoint`.

We can create a `HfInferenceEndpoint` providing our endpoint URL and credential.

```js
import { HfInferenceEndpoint } from '@huggingface/inference'

const hf = new HfInferenceEndpoint('https://YOUR_ENDPOINT.endpoints.huggingface.cloud', 'hf_YOUR_TOKEN')

// prompt
const prompt = 'What can you do in Nuremberg, Germany? Give me 3 Tips'

const stream = hf.textGenerationStream({ inputs: prompt })
for await (const r of stream) {
  // yield the generated token
  process.stdout.write(r.token.text)
}
```

## How does Streaming work under the hood?

Under the hood, TGI uses Server-Sent Events (SSE). In an SSE Setup, a client sends a request with the data, opening an HTTP connection and subscribing to updates. Afterward, the server sends data to the client. There is no need for further requests; the server will keep sending the data. SSEs are unidirectional, meaning the client does not send other requests to the server. SSE sends data over HTTP, making it easy to use.

SSEs are different than:
* Polling: where the client keeps calling the server to get data. This means that the server might return empty responses and cause overhead.
* Webhooks: where there is a bi-directional connection. The server can send information to the client, but the client can also send data to the server after the first request. Webhooks are more complex to operate as they don’t only use HTTP.

If there are too many requests at the same time, TGI returns an HTTP Error with an `overloaded` error type (`huggingface_hub` returns `OverloadedError`). This allows the client to manage the overloaded server (e.g., it could display a busy error to the user or retry with a new request). To configure the maximum number of concurrent requests, you can specify `--max_concurrent_requests`, allowing clients to handle backpressure.
