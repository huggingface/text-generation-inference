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

To enable the Messages API in Amazon SageMaker you need to set the environment variable `MESSAGES_API_ENABLED=true`. 

This will modify the `/invocations` route to accept Messages dictonaries consisting out of role and content. See the example below on how to deploy Llama with the new Messages API.

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