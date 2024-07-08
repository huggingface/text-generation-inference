# Quick Tour

The easiest way of getting started is using the official Docker container. Install Docker following [their installation instructions](https://docs.docker.com/get-docker/).

## Launching TGI

Let's say you want to deploy [teknium/OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) model with TGI on an Nvidia GPU. Here is an example on how to do that:

```bash
model=teknium/OpenHermes-2.5-Mistral-7B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.1.1 \
    --model-id $model
```

### Supported hardware

TGI supports various hardware. Make sure to check the [Using TGI with Nvidia GPUs](./installation_nvidia), [Using TGI with AMD GPUs](./installation_amd), [Using TGI with Intel GPUs](./installation_intel), [Using TGI with Gaudi](./installation_gaudi), [Using TGI with Inferentia](./installation_inferentia) guides depending on which hardware you would like to deploy TGI on.

## Consuming TGI

Once TGI is running, you can use the `generate` endpoint by doing requests. To learn more about how to query the endpoints, check the [Consuming TGI](./basic_tutorials/consuming_tgi) section, where we show examples with utility libraries and UIs. Below you can see a simple snippet to query the endpoint.

<inferencesnippet>
<python>

```python
import requests

headers = {
    "Content-Type": "application/json",
}

data = {
    'inputs': 'What is Deep Learning?',
    'parameters': {
        'max_new_tokens': 20,
    },
}

response = requests.post('http://127.0.0.1:8080/generate', headers=headers, json=data)
print(response.json())
# {'generated_text': '\n\nDeep Learning is a subset of Machine Learning that is concerned with the development of algorithms that can'}
```
</python>
<js>

```js
async function query() {
    const response = await fetch(
        'http://127.0.0.1:8080/generate',
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json'},
            body: JSON.stringify({
                'inputs': 'What is Deep Learning?',
                'parameters': {
                    'max_new_tokens': 20
                }
            })
        }
    );
}

query().then((response) => {
    console.log(JSON.stringify(response));
});
/// {"generated_text":"\n\nDeep Learning is a subset of Machine Learning that is concerned with the development of algorithms that can"}
```

</js>
<curl>

```curl
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

</curl>
</inferencesnippet>

<Tip>

To see all possible deploy flags and options, you can use the `--help` flag. It's possible to configure the number of shards, quantization, generation parameters, and more.

```bash
docker run ghcr.io/huggingface/text-generation-inference:2.1.1 --help
```

</Tip>
