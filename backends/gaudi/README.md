# Text-generation-inference - Gaudi backend

## Description

This is the TGI backend for Intel Gaudi. This backend is composed of the tgi server optimized for Gaudi hardware.

## Build your own image

The simplest way to build TGI with the Gaudi backend is to use the provided `Makefile`:

Option 1: From the project root directory:
```bash
make -C backends/gaudi image
```

Option 2: From the Gaudi backend directory:
```bash
cd backends/gaudi
make image
```

You can now run the server with the following command:

Option 1: Sharded:
```bash
model=meta-llama/Llama-3.1-8B-Instruct
hf_token=$(cat ${HOME}/.cache/huggingface/token)
volume=${HOME}/.cache/huggingface

docker run --runtime=habana --ipc=host --cap-add=sys_nice \
  -p 8080:80 -v $volume:/data \
  -e LOG_LEVEL=debug -e HF_TOKEN=$hf_token \
  tgi-gaudi --model-id $model \
  --sharded true --num-shard 8 \
  --max-input-tokens 512 --max-total-tokens 1024 --max-batch-size 8 --max-batch-prefill-tokens 2048
```

Option 2: Non-sharded:
```bash
model=meta-llama/Llama-3.1-8B-Instruct
hf_token=$(cat ${HOME}/.cache/huggingface/token)
volume=${HOME}/.cache/huggingface

docker run --runtime=habana --ipc=host --cap-add=sys_nice \
  -p 8080:80 -v $volume:/data \
  -e LOG_LEVEL=debug -e HF_TOKEN=$hf_token \
  tgi-gaudi --model-id $model \
  --max-input-tokens 512 --max-total-tokens 1024 --max-batch-size 4 --max-batch-prefill-tokens 2048
```

## Contributing

### Local Development

This is useful if you want to run the server locally for better debugging.
```bash
make -C backends/gaudi run-local-dev-container
```

Then run the following command inside the container to install tgi for gaudi:
```bash
make -C backends/gaudi local-dev-install
```

Add rust to path:
```bash
. "$HOME/.cargo/env"
```

Option 1: Run the server (sharded model):
```bash
LOG_LEVEL=debug text-generation-launcher \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --sharded true \
    --num-shard 8 \
    --max-input-tokens 512 \
    --max-total-tokens 1024 \
    --max-batch-size 8 \
    --max-batch-prefill-tokens 2048
```

Option 2: Run the server (non-sharded model):
```bash
LOG_LEVEL=debug text-generation-launcher \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --max-input-tokens 512 \
    --max-total-tokens 1024 \
    --max-batch-size 4 \
    --max-batch-prefill-tokens 2048
```

You can then test the server with the following curl command from another terminal (can be outside the container):
```bash
curl 127.0.0.1:8080/generate \
     -X POST \
     -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
     -H 'Content-Type: application/json'
```

### Integration tests

To run the integration tests, you need to first build the image:
```bash
make -C backends/gaudi image
```

Then run the following command to run the integration tests:
```bash
make -C backends/gaudi run-integration-tests
```

To capture the expected outputs for the integration tests, you can run the following command:
```bash
make -C backends/gaudi capture-expected-outputs-for-integration-tests
```

#### How the integration tests works
The integration tests works as follows:

1. Start a tgi server in a container, similar to the command:
```bash
docker run --runtime=habana --ipc=host --cap-add=sys_nice \
  -p 8080:80 -v $volume:/data \
  -e LOG_LEVEL=debug -e HF_TOKEN=$hf_token \
  tgi-gaudi --model-id $model \
  --max-input-tokens 512 --max-total-tokens 1024 --max-batch-size 4 --max-batch-prefill-tokens 2048
```

2. Do a /generate request to the server, similar to the command:
```bash
curl 127.0.0.1:8080/generate \
     -X POST \
     -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
     -H 'Content-Type: application/json'
```

3. Check the output of the server against the expected output:
```python
assert curl_output == expected_output
```

This is the repeated for a set of models and configurations.
