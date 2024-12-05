# TensorRT-LLM backend

The NVIDIA TensorRT-LLM (TRTLLM) backend is a high-performance backend for LLMs
that uses NVIDIA's TensorRT library for inference acceleration.
It makes use of specific optimizations for NVIDIA GPUs, such as custom kernels.

To use the TRTLLM backend you need to compile `engines` for the models you want to use.
Each `engine` must be compiled on the same GPU architecture that you will use for inference.

## Supported models

Check the [support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) to see which models are
supported.

## Compiling engines

You can use [Optimum-NVIDIA](https://github.com/huggingface/optimum-nvidia) to compile engines for the models you
want to use.

```bash 
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Install huggingface_cli
python -m pip install huggingface-cli[hf_transfer]

# Login to the Hugging Face Hub
huggingface-cli login

# Create a directory to store the model
mkdir -p /tmp/models/$MODEL_NAME

# Create a directory to store the compiled engine
mkdir -p /tmp/engines/$MODEL_NAME

# Download the model 
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --local-dir /tmp/models/$MODEL_NAME $MODEL_NAME

# Compile the engine using Optimum-NVIDIA
docker run \
  --rm \
  -it \
  --gpus=1 \
  -v /tmp/models/$MODEL_NAME:/model \
  -v /tmp/engines/$MODEL_NAME:/engine \
  huggingface/optimum-nvidia \
    optimum-cli export trtllm \
    --tp=1 \
    --pp=1 \
    --max-batch-size=128 \
    --max-input-length 4096 \
    --max-output-length 8192 \
    --max-beams-width=1 \
    --destination /engine \
    $MODEL_NAME
```

Your compiled engine will be saved in the `/tmp/engines/$MODEL_NAME` directory.

## Using the TRTLLM backend

Run TGI-TRTLLM Docker image with the compiled engine:

```bash
docker run \
  --gpus 1 \
  -it \
  --rm \
  -p 3000:3000 \
  -e MODEL=$MODEL_NAME \
  -e PORT=3000 \
  -e HF_TOKEN='hf_XXX' \
  -v /tmp/engines/$MODEL_NAME:/data \ 
  ghcr.io/huggingface/text-generation-inference:latest-trtllm \
  --executor-worker executorWorker \
  --model-id /data/$MODEL_NAME
```

## Development

To develop TRTLLM backend, you can use [dev containers](https://containers.dev/) located in
`.devcontainer` directory.