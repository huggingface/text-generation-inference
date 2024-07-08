# Consuming Text Generation Inference

There are many ways you can consume Text Generation Inference server in your applications. After launching, you can use the `/generate` route and make a `POST` request to get results from the server. You can also use the `/generate_stream` route if you want TGI to return a stream of tokens. You can make the requests using the tool of your preference, such as curl, Python or TypeScrpt. For a final end-to-end experience, we also open-sourced ChatUI, a chat interface for open-source models.

## curl

After the launch, you can query the model using either the `/generate` or `/generate_stream` routes:

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```


## Inference Client

[`huggingface-hub`](https://huggingface.co/docs/huggingface_hub/main/en/index) is a Python library to interact with the Hugging Face Hub, including its endpoints. It provides a nice high-level class, [`~huggingface_hub.InferenceClient`], which makes it easy to make calls to a TGI endpoint. `InferenceClient` also takes care of parameter validation and provides a simple to-use interface.
You can simply install `huggingface-hub` package with pip.

```bash
pip install huggingface-hub
```

Once you start the TGI server, instantiate `InferenceClient()` with the URL to the endpoint serving the model. You can then call `text_generation()` to hit the endpoint through Python.

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://127.0.0.1:8080")
client.text_generation(prompt="Write a code for snake game")
```

You can do streaming with `InferenceClient` by passing `stream=True`. Streaming will return tokens as they are being generated in the server. To use streaming, you can do as follows:

```python
for token in client.text_generation("How do you make cheese?", max_new_tokens=12, stream=True):
    print(token)
```

Another parameter you can use with TGI backend is `details`. You can get more details on generation (tokens, probabilities, etc.) by setting `details` to `True`. When it's specified, TGI will return a `TextGenerationResponse` or `TextGenerationStreamResponse` rather than a string or stream.

```python
output = client.text_generation(prompt="Meaning of life is", details=True)
print(output)

# TextGenerationResponse(generated_text=' a complex concept that is not always clear to the individual. It is a concept that is not always', details=Details(finish_reason=<FinishReason.Length: 'length'>, generated_tokens=20, seed=None, prefill=[], tokens=[Token(id=267, text=' a', logprob=-2.0723474, special=False), Token(id=11235, text=' complex', logprob=-3.1272552, special=False), Token(id=17908, text=' concept', logprob=-1.3632495, special=False),..))
```

You can see how to stream below.

```python
output = client.text_generation(prompt="Meaning of life is", stream=True, details=True)
print(next(iter(output)))

# TextGenerationStreamResponse(token=Token(id=267, text=' a', logprob=-2.0723474, special=False), generated_text=None, details=None)
```

You can check out the details of the function [here](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_generation). There is also an async version of the client, `AsyncInferenceClient`, based on `asyncio` and `aiohttp`. You can find docs for it [here](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)


## ChatUI

ChatUI is an open-source interface built for LLM serving. It offers many customization options, such as web search with SERP API and more. ChatUI can automatically consume the TGI server and even provides an option to switch between different TGI endpoints. You can try it out at [Hugging Chat](https://huggingface.co/chat/), or use the [ChatUI Docker Space](https://huggingface.co/new-space?template=huggingchat/chat-ui-template) to deploy your own Hugging Chat to Spaces.

To serve both ChatUI and TGI in same environment, simply add your own endpoints to the `MODELS` variable in `.env.local` file inside the `chat-ui` repository. Provide the endpoints pointing to where TGI is served.

```
{
// rest of the model config here
"endpoints": [{"url": "https://HOST:PORT/generate_stream"}]
}
```

![ChatUI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/chatui_screen.png)

## Gradio

Gradio is a Python library that helps you build web applications for your machine learning models with a few lines of code. It has a `ChatInterface` wrapper that helps create neat UIs for chatbots. Let's take a look at how to create a chatbot with streaming mode using TGI and Gradio. Let's install Gradio and Hub Python library first.

```bash
pip install huggingface-hub gradio
```

Assume you are serving your model on port 8080, we will query through [InferenceClient](consuming_tgi#inference-client).

```python
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://127.0.0.1:8080")

def inference(message, history):
    partial_message = ""
    for token in client.text_generation(message, max_new_tokens=20, stream=True):
        partial_message += token
        yield partial_message

gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
    description="This is the demo for Gradio UI consuming TGI endpoint with LLaMA 7B-Chat model.",
    title="Gradio 🤝 TGI",
    examples=["Are tomatoes vegetables?"],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
).queue().launch()
```

The UI looks like this 👇

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/gradio-tgi.png"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/gradio-tgi-dark.png"
    />
</div>

You can try the demo directly here 👇

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


You can disable streaming mode using `return` instead of `yield` in your inference function, like below.

```python
def inference(message, history):
    return client.text_generation(message, max_new_tokens=20)
```

You can read more about how to customize a `ChatInterface` [here](https://www.gradio.app/guides/creating-a-chatbot-fast).

## API documentation

You can consult the OpenAPI documentation of the `text-generation-inference` REST API using the `/docs` route. The Swagger UI is also available [here](https://huggingface.github.io/text-generation-inference).
