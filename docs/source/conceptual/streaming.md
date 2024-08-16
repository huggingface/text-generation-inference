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

client = InferenceClient(base_url="http://127.0.0.1:8080")
output = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)

# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
```

The `huggingface_hub` library also comes with an `AsyncInferenceClient` in case you need to handle the requests concurrently.

```python
from huggingface_hub import AsyncInferenceClient

client = AsyncInferenceClient(base_url="http://127.0.0.1:8080")
async def main():
    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

asyncio.run(main())

# This
# is
# a
# test
#.
```

### Streaming with cURL

To use the OpenAI Chat Completions compatible Messages API `v1/chat/completions` endpoint with curl, you can add the `-N` flag, which disables curl default buffering and shows data as it arrives from the server

```curl
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": true,
  "max_tokens": 20
}' \
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
