# Text-generation-inference - Gaudi backend

## Description

This is the TGI backend for Intel Gaudi. This backend is composed of the tgi server optimized for Gaudi hardware.

## Build your own image

The simplest way to build TGI with the gaudi backend is to use the provided `Makefile`:

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
```bash
model=meta-llama/Llama-3.1-8B-Instruct
hf_token=$(cat ${HOME}/.cache/huggingface/token)
volume=${HOME}/.cache/huggingface

docker run -p 8080:80 -v $volume:/data --runtime=habana -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
-e LOG_LEVEL=debug \
-e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
-e HF_TOKEN=$hf_token -e ENABLE_HPU_GRAPH=true -e LIMIT_HPU_GRAPH=true \
-e USE_FLASH_ATTENTION=true -e FLASH_ATTENTION_RECOMPUTE=true --cap-add=sys_nice \
--ipc=host tgi-gaudi --model-id $model --sharded true \
--num-shard 8 --max-input-tokens 512 --max-total-tokens 1024 --max-batch-size 8 --max-batch-prefill-tokens 2048 --max-batch-total-tokens 8192
```

## Contributing

### Local Development

This is useful if you want to run the server in locally for better debugging.
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
