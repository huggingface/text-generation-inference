# Consuming Text Generation Inference

There are many ways to consume Text Generation Inference (TGI) server in your applications. After launching the server, you can use the [Messages API](https://huggingface.co/docs/text-generation-inference/en/messages_api) `/v1/chat/completions` route and make a `POST` request to get results from the server. You can also pass `"stream": true` to the call if you want TGI to return a stream of tokens.

For more information on the API, consult the OpenAPI documentation of `text-generation-inference` available [here](https://huggingface.github.io/text-generation-inference).

You can make the requests using any tool of your preference, such as curl, Python, or TypeScript. For an end-to-end experience, we've open-sourced [ChatUI](https://github.com/huggingface/chat-ui), a chat interface for open-access models.

## curl

After a successful server launch, you can query the model using the `v1/chat/completions` route, to get responses that are compliant to the OpenAI Chat Completion spec:

```bash
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

For non-chat use-cases, you can also use the `/generate` and `/generate_stream` routes.

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{
  "inputs":"What is Deep Learning?",
  "parameters":{
    "max_new_tokens":20
  }
}' \
    -H 'Content-Type: application/json'
```

## Python

### Inference Client

[`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/main/en/index) is a Python library to interact with the Hugging Face Hub, including its endpoints. It provides a high-level class, [`huggingface_hub.InferenceClient`](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient), which makes it easy to make calls to TGI's Messages API. `InferenceClient` also takes care of parameter validation and provides a simple-to-use interface.

Install `huggingface_hub` package via pip.

```bash
pip install huggingface_hub
```

You can now use `InferenceClient` the exact same way you would use `OpenAI` client in Python

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    base_url="http://localhost:8080/v1/",
)

output = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
```

You can check out more details about OpenAI compatibility [here](https://huggingface.co/docs/huggingface_hub/en/guides/inference#openai-compatibility).

There is also an async version of the client, `AsyncInferenceClient`, based on `asyncio` and `aiohttp`. You can find docs for it [here](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)

### OpenAI Client

You can directly use the OpenAI [Python](https://github.com/openai/openai-python) or [JS](https://github.com/openai/openai-node) clients to interact with TGI.

Install the OpenAI Python package via pip.

```bash
pip install openai
```

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="http://localhost:8080/v1/",
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

## UI

### Gradio

Gradio is a Python library that helps you build web applications for your machine learning models with a few lines of code. It has a `ChatInterface` wrapper that helps create neat UIs for chatbots. Let's take a look at how to create a chatbot with streaming mode using TGI and Gradio. Let's install Gradio and Hub Python library first.

```bash
pip install huggingface-hub gradio
```

Assume you are serving your model on port 8080, we will query through [InferenceClient](consuming_tgi#inference-client).

```python
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(base_url="http://127.0.0.1:8080")

def inference(message, history):
    partial_message = ""
    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        stream=True,
        max_tokens=1024,
    )

    for chunk in output:
        partial_message += chunk.choices[0].delta.content
        yield partial_message

gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
    description="This is the demo for Gradio UI consuming TGI endpoint.",
    title="Gradio 🤝 TGI",
    examples=["Are tomatoes vegetables?"],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
).queue().launch()
```

You can check out the UI and try the demo directly here 👇

<div class="block dark:hidden">
	<iframe
        src="https://merve-gradio-tgi-2.hf.space?__theme=light"
        width="850"
        height="750"
    ></iframe>
</div>
<div class="hidden dark:block">
    <iframe
        src="https://merve-gradio-tgi-2.hf.space?__theme=dark"
        width="850"
        height="750"
    ></iframe>
</div>


You can read more about how to customize a `ChatInterface` [here](https://www.gradio.app/guides/creating-a-chatbot-fast).

### ChatUI

[ChatUI](https://github.com/huggingface/chat-ui) is an open-source interface built for consuming LLMs. It offers many customization options, such as web search with SERP API and more. ChatUI can automatically consume the TGI server and even provides an option to switch between different TGI endpoints. You can try it out at [Hugging Chat](https://huggingface.co/chat/), or use the [ChatUI Docker Space](https://huggingface.co/new-space?template=huggingchat/chat-ui-template) to deploy your own Hugging Chat to Spaces.

To serve both ChatUI and TGI in same environment, simply add your own endpoints to the `MODELS` variable in `.env.local` file inside the `chat-ui` repository. Provide the endpoints pointing to where TGI is served.

```
{
// rest of the model config here
"endpoints": [{"url": "https://HOST:PORT/generate_stream"}]
}
```

![ChatUI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/chatui_screen.png)
