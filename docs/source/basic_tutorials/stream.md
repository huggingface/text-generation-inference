# Stream responses in Javascript and Python

Requesting and generating text with LLMs can be a time-consuming and iterative process. A great way to improve the user experience is streaming tokens to the user as they are generated. Below are two examples of how to stream tokens using Python and JavaScript. For Python, we are going to use the **[huggingface_hub library](https://huggingface.co/docs/huggingface_hub/index), and for JavaScript, the [HuggingFace.js library](https://huggingface.co/docs/huggingface.js/main/en/index)

## Streaming requests with Python

First, you need to install the `huggingface_hub` library:

`pip install -U huggingface_hub`

We can create a `InferenceClient` providing our endpoint URL and credential alongside the hyperparameters we want to use

```python
from huggingface_hub import InferenceClient

# HF Inference Endpoints parameter
endpoint_url = "https://YOUR_ENDPOINT.endpoints.huggingface.cloud"
hf_token = "hf_YOUR_TOKEN"

# Streaming Client
client = InferenceClient(endpoint_url, token=hf_token)

# generation parameter
gen_kwargs = dict(
    max_new_tokens=512,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repetition_penalty=1.02,
    stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
)
# prompt
prompt = "What can you do in Nuremberg, Germany? Give me 3 Tips"

stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)

# yield each generated token
for r in stream:
    # skip special tokens
    if r.token.special:
        continue
    # stop if we encounter a stop sequence
    if r.token.text in gen_kwargs["stop_sequences"]:
        break
    # yield the generated token
    print(r.token.text, end = "")
    # yield r.token.text
```

Replace the `print` command with the `yield` or with a function you want to stream the tokens to.

## Streaming requests with JavaScript

First, you need to install the `@huggingface/inference` library.

`npm install @huggingface/inference`

We can create a `HfInferenceEndpoint` providing our endpoint URL and credential alongside the hyperparameter we want to use.

```js
import { HfInferenceEndpoint } from '@huggingface/inference'

const hf = new HfInferenceEndpoint('https://YOUR_ENDPOINT.endpoints.huggingface.cloud', 'hf_YOUR_TOKEN')

//generation parameter
const gen_kwargs = {
  max_new_tokens: 512,
  top_k: 30,
  top_p: 0.9,
  temperature: 0.2,
  repetition_penalty: 1.02,
  stop_sequences: ['\nUser:', '<|endoftext|>', '</s>'],
}
// prompt
const prompt = 'What can you do in Nuremberg, Germany? Give me 3 Tips'

const stream = hf.textGenerationStream({ inputs: prompt, parameters: gen_kwargs })
for await (const r of stream) {
  // # skip special tokens
  if (r.token.special) {
    continue
  }
  // stop if we encounter a stop sequence
  if (gen_kwargs['stop_sequences'].includes(r.token.text)) {
    break
  }
  // yield the generated token
  process.stdout.write(r.token.text)
}
```

Replace the `process.stdout` call with the `yield` or with a function you want to stream the tokens to.