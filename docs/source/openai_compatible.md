# Messages API

_Messages API is compatible to OpenAI Chat Completion API_

Text Generation Inference (TGI) now supports the Message API which is fully compatible with the OpenAI Chat Completion API. This means you can use OpenAI's client libraries to interact with TGI's Messages API. Below are some examples of how to utilize this compatibility.

## Making a Request

You can make a request to TGI's Messages API using `curl`. Here's an example:

```bash
curl localhost:3000/v1/chat/completions \
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

## Streaming

You can also use OpenAI's Python client library to make a streaming request. Here's how:

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="-"
)

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is deep learning?"}
    ],
    stream=True
)

# iterate and print stream
for message in chat_completion:
    print(message)
```

## Synchronous

If you prefer to make a synchronous request, you can do so like this:

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="-"
)

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is deep learning?"}
    ],
    stream=False
)

print(chat_completion)
```

## Cloud Providers

TGI can be deployed on various cloud providers for scalable and robust text generation. One such provider is Amazon SageMaker, which has recently added support for TGI. Here's how you can deploy TGI on Amazon SageMaker:

## Amazon SageMaker

Amazon SageMaker allows two routes: `/invocations` and `/ping` (or `/health`) for health checks. By default, we map `/generate` to `/invocations`. However, SageMaker does not allow requests to any other routes. 

To provide the new feature of Messages API, we have introduced an environment variable `OAI_ENABLED`. If `OAI_ENABLED=true`, the `chat_completions` method is used when `/invocations` is called, otherwise it defaults to `generate`. This allows users to opt in for the OAI format.

Here's an example of running the router with `OAI_ENABLED` set to `true`:

```bash
OAI_ENABLED=true text-generation-launcher --model-id <MODEL-ID>
```

And here's an example request:

```bash
curl <SAGEMAKER-ENDPOINT>/invocations \
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
    -H 'Content-Type: application/json' | jq 
```

Please let us know if any naming changes are needed or if any other routes need similar functionality.
