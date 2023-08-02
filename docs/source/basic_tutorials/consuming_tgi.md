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