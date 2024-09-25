# Vision Language Model Inference in TGI

Visual Language Model (VLM) are models that consume both image and text inputs to generate text.

VLM's are trained on a combination of image and text data and can handle a wide range of tasks, such as image captioning, visual question answering, and visual dialog.

> What distinguishes VLMs from other text and image models is their ability to handle long context and generate text that is coherent and relevant to the image even after multiple turns or in some cases, multiple images.

Below are couple of common use cases for vision language models:

- **Image Captioning**: Given an image, generate a caption that describes the image.
- **Visual Question Answering (VQA)**: Given an image and a question about the image, generate an answer to the question.
- **Mulimodal Dialog**: Generate response to multiple turns of images and conversations.
- **Image Information Retrieval**: Given an image, retrieve information from the image.

## How to Use a Vision Language Model?

### Hugging Face Hub Python Library

To infer with vision language models through Python, you can use the [`huggingface_hub`](https://pypi.org/project/huggingface-hub/) library. The `InferenceClient` class provides a simple way to interact with the [Inference API](https://huggingface.co/docs/api-inference/index). Images can be passed as URLs or base64-encoded strings. The `InferenceClient` will automatically detect the image format.

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://127.0.0.1:3000")
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
prompt = f"![]({image})What is this a picture of?\n\n"
for token in client.text_generation(prompt, max_new_tokens=16, stream=True):
    print(token)

# This is a picture of an anthropomorphic rabbit in a space suit.
```

```python
from huggingface_hub import InferenceClient
import base64
import requests
import io

client = InferenceClient("http://127.0.0.1:3000")

# read image from local file
image_path = "rabbit.png"
with open(image_path, "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")

image = f"data:image/png;base64,{image}"
prompt = f"![]({image})What is this a picture of?\n\n"

for token in client.text_generation(prompt, max_new_tokens=10, stream=True):
    print(token)

# This is a picture of an anthropomorphic rabbit in a space suit.
```

or via the `chat_completion` endpoint:

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://127.0.0.1:3000")

chat = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Whats in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
                    },
                },
            ],
        },
    ],
    seed=42,
    max_tokens=100,
)

print(chat)
# ChatCompletionOutput(choices=[ChatCompletionOutputComplete(finish_reason='length', index=0, message=ChatCompletionOutputMessage(role='assistant', content=" The image you've provided features an anthropomorphic rabbit in spacesuit attire. This rabbit is depicted with human-like posture and movement, standing on a rocky terrain with a vast, reddish-brown landscape in the background. The spacesuit is detailed with mission patches, circuitry, and a helmet that covers the rabbit's face and ear, with an illuminated red light on the chest area.\n\nThe artwork style is that of a", name=None, tool_calls=None), logprobs=None)], created=1714589614, id='', model='llava-hf/llava-v1.6-mistral-7b-hf', object='text_completion', system_fingerprint='2.0.2-native', usage=ChatCompletionOutputUsage(completion_tokens=100, prompt_tokens=2943, total_tokens=3043))

```

or with OpenAI's [client library](https://github.com/openai/openai-python):

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(base_url="http://localhost:3000/v1", api_key="-")

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Whats in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
                    },
                },
            ],
        },
    ],
    stream=False,
)

print(chat_completion)
# ChatCompletion(id='', choices=[Choice(finish_reason='eos_token', index=0, logprobs=None, message=ChatCompletionMessage(content=' The image depicts an anthropomorphic rabbit dressed in a space suit with gear that resembles NASA attire. The setting appears to be a solar eclipse with dramatic mountain peaks and a partial celestial body in the sky. The artwork is detailed and vivid, with a warm color palette and a sense of an adventurous bunny exploring or preparing for a journey beyond Earth. ', role='assistant', function_call=None, tool_calls=None))], created=1714589732, model='llava-hf/llava-v1.6-mistral-7b-hf', object='text_completion', system_fingerprint='2.0.2-native', usage=CompletionUsage(completion_tokens=84, prompt_tokens=2943, total_tokens=3027))
```

### Inference Through Sending `cURL` Requests

To use the `generate_stream` endpoint with curl, you can add the `-N` flag. This flag disables curl default buffering and shows data as it arrives from the server.

```bash
curl -N 127.0.0.1:3000/generate_stream \
    -X POST \
    -d '{"inputs":"![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)What is this a picture of?\n\n","parameters":{"max_new_tokens":16, "seed": 42}}' \
    -H 'Content-Type: application/json'

# ...
# data:{"index":16,"token":{"id":28723,"text":".","logprob":-0.6196289,"special":false},"generated_text":"This is a picture of an anthropomorphic rabbit in a space suit.","details":null}
```

### Inference Through JavaScript

First, we need to install the `@huggingface/inference` library.

```bash
npm install @huggingface/inference
```

If you're using the free Inference API, you can use [Huggingface.js](https://huggingface.co/docs/huggingface.js/inference/README)'s `HfInference`. If you're using inference endpoints, you can use `HfInferenceEndpoint` class to easily interact with the Inference API.

We can create a `HfInferenceEndpoint` providing our endpoint URL and We can create a `HfInferenceEndpoint` providing our endpoint URL and [Hugging Face access token](https://huggingface.co/settings/tokens).

```js
import { HfInferenceEndpoint } from "@huggingface/inference";

const hf = new HfInferenceEndpoint("http://127.0.0.1:3000", "HF_TOKEN");

const prompt =
  "![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)What is this a picture of?\n\n";

const stream = hf.textGenerationStream({
  inputs: prompt,
  parameters: { max_new_tokens: 16, seed: 42 },
});
for await (const r of stream) {
  // yield the generated token
  process.stdout.write(r.token.text);
}

// This is a picture of an anthropomorphic rabbit in a space suit.
```

## Combining Vision Language Models with Other Features

VLMs in TGI have several advantages, for example these models can be used in tandem with other features for more complex tasks. For example, you can use VLMs with [Guided Generation](/docs/conceptual/guided-generation) to generate specific JSON data from an image.

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
        width="400"
    />
</div>

For example we can extract information from the rabbit image and generate a JSON object with the location, activity, number of animals seen, and the animals seen. That would look like this:

```json
{
  "activity": "Standing",
  "animals": ["Rabbit"],
  "animals_seen": 1,
  "location": "Rocky surface with mountains in the background and a red light on the rabbit's chest"
}
```

All we need to do is provide a JSON schema to the VLM model and it will generate the JSON object for us.

```bash
curl localhost:3000/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
    "inputs":"![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)What is this a picture of?\n\n",
    "parameters": {
        "max_new_tokens": 100,
        "seed": 42,
        "grammar": {
            "type": "json",
            "value": {
                "properties": {
                    "location": {
                        "type": "string"
                    },
                    "activity": {
                        "type": "string"
                    },
                    "animals_seen": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "animals": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["location", "activity", "animals_seen", "animals"]
            }
        }
    }
}'

# {
#   "generated_text": "{ \"activity\": \"Standing\", \"animals\": [ \"Rabbit\" ], \"animals_seen\": 1, \"location\": \"Rocky surface with mountains in the background and a red light on the rabbit's chest\" }"
# }
```

Want to learn more about how Vision Language Models work? Check out the [awesome blog post on the topic](https://huggingface.co/blog/vlms).
