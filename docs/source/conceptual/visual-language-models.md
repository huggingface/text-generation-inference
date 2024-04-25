# Vision Language Models (VLM)

## What is VLM?

Visual Language Model (VLM) are models that consume both visual and textual inputs to generate text.

These models are trained on multimodal data, which includes both images and text.

VLMs can be used for a variety of tasks, such as image captioning, visual question answering, and more.

<div class="flex justify-center">
    <pre>placeholder for architecture diagram</pre>
</div>

With VLM, you can generate text from an image. For example, you can generate a caption for an image, answer questions about an image, or generate a description of an image.

- **Image Captioning**: Given an image, generate a caption that describes the image.
- **Visual Question Answering (VQA)**: Given an image and a question about the image, generate an answer to the question.
- **Visual Dialog**: Given an image and a dialog history, generate a response to the dialog.
- **Visual Data Extraction**: Given an image, extract information from the image.

For example, given the image of a cat, a VLM can generate the caption "A cat sitting on a couch" or answer the question "What is the cat doing?" with "The cat is sitting on a couch."

## How to use VLM?

### VLM with Python

To use VLM with Python, you can use the `huggingface_hub` library. The `InferenceClient` class provides a simple way to interact with the Inference API.

This is the following image:

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
        width="400"
    />
</div>

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://127.0.0.1:3000")
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
prompt = f"![]({image})What is this a picture of?\n\n"
for token in client.text_generation(prompt, max_new_tokens=16, stream=True):
    print(token)

# This
#  is
#  a
#  picture
#  of
#  an
#  anth
# rop
# omorphic
#  rab
# bit
#  in
#  a
#  space
#  suit
# .
```

Images can be passed as URLs or base64-encoded strings. The `InferenceClient` will automatically detect the image format.

This is the following image:

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png"
        width="400"
    />
</div>

```python
from huggingface_hub import InferenceClient
import base64
import requests
import io

client = InferenceClient("http://127.0.0.1:3000")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png"
original_image = requests.get(url)

# encode image to base64
image_bytes = io.BytesIO(original_image.content)
image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
image = f"data:image/png;base64,{image}"

prompt = f"![]({image})What is this a picture of?\n\n"

for token in client.text_generation(prompt, max_new_tokens=10, stream=True):
    print(token)

# This
#  is
#  a
#  picture
#  of
#  a
#  be
# aver
# .
```

If you want additional details, you can add `details=True`. In this case, you get a `TextGenerationStreamResponse` which contains additional information such as the probabilities and the tokens. For the final response in the stream, it also returns the full generated text.

### VLM with cURL

To use the `generate_stream` endpoint with curl, you can add the `-N` flag, which disables curl default buffering and shows data as it arrives from the server

```bash
curl -N 127.0.0.1:3000/generate_stream \
    -X POST \
    -d '{"inputs":"![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)What is this a picture of?\n\n","parameters":{"max_new_tokens":16, "seed": 42}}' \
    -H 'Content-Type: application/json'

# ...
# data:{"index":16,"token":{"id":28723,"text":".","logprob":-0.6196289,"special":false},"generated_text":"This is a picture of an anthropomorphic rabbit in a space suit.","details":null}
```

### VLM with JavaScript

First, we need to install the `@huggingface/inference` library.
`npm install @huggingface/inference`

If you're using the free Inference API, you can use `HfInference`. If you're using inference endpoints, you can use `HfInferenceEndpoint`.

We can create a `HfInferenceEndpoint` providing our endpoint URL and credential.

```js
import { HfInferenceEndpoint } from "@huggingface/inference";

const hf = new HfInferenceEndpoint("http://127.0.0.1:3000", "hf_YOUR_TOKEN");

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

## Advantages of VLM in TGI

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

## How does VLM work under the hood?

coming soon...

<pre>placeholder for architecture diagram (image to tokens)</pre>
