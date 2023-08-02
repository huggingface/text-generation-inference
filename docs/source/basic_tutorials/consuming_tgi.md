# Consuming Text Generation Inference

## ChatUI

ChatUI is the open-source interface built for large language model serving. It offers many customization options, web search with SERP API and more. ChatUI can automatically consume the Text Generation Inference server, and even provide option to switch between different TGI endpoints. You can try it out at [Hugging Chat](https://huggingface.co/chat/), or use [ChatUI Docker Spaces](https://huggingface.co/new-space?template=huggingchat/chat-ui-template) to deploy your own Hugging Chat to Spaces.

To serve both ChatUI and TGI in same environment, simply add your own endpoints to the `MODELS` variable in ``.env.local` file inside `chat-ui` repository. Provide the endpoints pointing to where TGI is served.

```
{
// rest of the model config here
"endpoints": [{"url": "https://HOST:PORT/generate_stream"}]
}
```

## Inference Client

`huggingface-hub` is a Python library to interact and manage repositories and endpoints on Hugging Face Hub. `InferenceClient` is a class that lets users interact with models on Hugging Face Hub and Hugging Face models served by any TGI endpoint. Once you start the TGI server, simply instantiate `InferenceClient()` with the URL to endpoint serving the model. You can then call `text_generation()` to hit the endpoint through Python. 

```python
from huggingface_hub import InferenceClient
client = InferenceClient(model=URL_TO_ENDPOINT_SERVING_TGI)
client.text_generation(prompt="Write a code for snake game", model=URL_TO_ENDPOINT_SERVING_TGI)
```

You can check out the details of the function [here](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_generation).