# HTTP API Reference

#### Table of Contents

- [Text Generation Inference custom API](#text-generation-inference-custom-api)
- [OpenAI Messages API](#openai-messages-api)
  - [Making a Request](#making-a-request)
  - [Streaming](#streaming)
  - [Synchronous](#synchronous)
  - [Hugging Face Inference Endpoints](#hugging-face-inference-endpoints)
  - [Cloud Providers](#cloud-providers)
    - [Amazon SageMaker](#amazon-sagemaker)
    - [Google Cloud](#google-cloud)

The HTTP API is a RESTful API that allows you to interact with the text-generation-inference component. Two endpoints are available:

- Text Generation Inference [custom API](https://huggingface.github.io/text-generation-inference/)
- OpenAI's [Messages API](#openai-messages-api)

## Text Generation Inference custom API

Check the [API documentation](https://huggingface.github.io/text-generation-inference/) for more information on how to interact with the Text Generation Inference API.

## OpenAI Messages API

Text Generation Inference (TGI) now supports the Messages API, which is fully compatible with the OpenAI Chat Completion API. This feature is available starting from version 1.4.0. You can use OpenAI's client libraries or third-party libraries expecting OpenAI schema to interact with TGI's Messages API. Below are some examples of how to utilize this compatibility.

> **Note:** The Messages API is supported from TGI version 1.4.0 and above. Ensure you are using a compatible version to access this feature.

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

## Hugging Face Inference Endpoints

The Messages API is integrated with [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated).
Every endpoint that uses "Text Generation Inference" with an LLM, which has a chat template can now be used. Below is an example of how to use IE with TGI using OpenAI's Python client library:

> **Note:** Make sure to replace `base_url` with your endpoint URL and to include `v1/` at the end of the URL. The `api_key` should be replaced with your Hugging Face API key.

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    # replace with your endpoint url, make sure to include "v1/" at the end
    base_url="https://vlzz10eq3fol3429.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    # replace with your API key
    api_key="hf_XXX"
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
    print(message.choices[0].delta.content, end="")
```

## Cloud Providers

TGI can be deployed on various cloud providers for scalable and robust text generation. Among those cloud providers, both Amazon Sagemaker and Google Cloud have TGI integrations within their cloud offering.

### Amazon SageMaker

To enable the Messages API in Amazon SageMaker you need to set the environment variable `MESSAGES_API_ENABLED=true`.

This will modify the `/invocations` route to accept Messages dictionaries consisting out of role and content. See the example below on how to deploy Llama with the new Messages API.

```python
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {
    'HF_MODEL_ID':'HuggingFaceH4/zephyr-7b-beta',
    'SM_NUM_GPUS': json.dumps(1),
    'MESSAGES_API_ENABLED': True
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface",version="1.4.0"),
    env=hub,
    role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
)

# send request
predictor.predict({
    "messages": [
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is deep learning?"}
    ]
})
```

### Google Cloud

A collection of publicly available Deep Learning Containers (DLCs) are available for TGI on Google Cloud for services such as Google Kubernetes Engine (GKE), Vertex AI or Cloud Run.

The TGI DLCs are built with the `--features google` flag as well as including the Google SDK installation, in order to better coexist within the Google Cloud environment, whilst being seamlessly integrated with Vertex AI supporting the custom I/O formatting.

The DLCs are publicly available on the [Google Cloud Deep Learning Containers Documentation for TGI](https://cloud.google.com/deep-learning-containers/docs/choosing-container#text-generation-inference), the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or use the `gcloud` command to list the available containers with the tag `huggingface-text-generation-inference` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-text-generation-inference"
```

The containers can be used within any Google Cloud service, you can find some examples below:

- [Deploy Meta Llama 3 8B with TGI DLC on GKE](https://github.com/huggingface/Google-Cloud-Containers/tree/main/examples/gke/tgi-deployment)
- [Deploy Gemma 7B with TGI DLC on Vertex AI](https://github.com/huggingface/Google-Cloud-Containers/blob/main/examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai/vertex-notebook.ipynb)
- [Deploy Meta Llama 3.1 8B with TGI DLC on Cloud Run](https://github.com/huggingface/Google-Cloud-Containers/tree/main/examples/cloud-run/tgi-deployment)

More information and examples available in [the Google-Cloud-Containers repository](https://github.com/huggingface/Google-Cloud-Containers).
