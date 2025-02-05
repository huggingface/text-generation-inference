# Llamacpp backend

The llamacpp backend is a backend for running LLMs using the `llama.cpp`
project. It supports CPU and GPU inference and is easy to deploy without
complex dependencies. For more details, visit the official repository:
[llama.cpp](https://github.com/ggerganov/llama.cpp).

## Supported models

`llama.cpp` uses the GGUF format, which supports various quantization
levels to optimize performance and reduce memory usage. Learn more and
find GGUF models on [Hugging Face](https://huggingface.co/models?search=gguf).

## Building the Docker image

The llamacpp backend is optimized for the local machine, so it is highly
recommended to build the Docker image on the same machine where it will
be used for inference. You can build it directly from the GitHub
repository without cloning using the following command:

```bash
docker build \
    -t llamacpp-backend \
    https://github.com/huggingface/text-generation-inference.git \
    -f Dockerfile_llamacpp
```

### Build arguments

You can customize the build using the following arguments:

| Argument                               | Description                                  |
|----------------------------------------|----------------------------------------------|
| `--build-arg llamacpp_version=VERSION` | Specifies a particular version of llama.cpp. |
| `--build-arg llamacpp_cuda=ON`         | Enables CUDA support.                        |
| `--build-arg cuda_arch=ARCH`           | Selects the target GPU architecture.         |

## Preparing the model

Before running TGI, you need a GGUF model, for example:

```bash
mkdir -p ~/models
cd ~/models
curl -O "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf?download=true"
```

## Running the llamacpp backend

Run TGI with the llamacpp backend and your chosen model. When using GPU
inference, you need to set `--gpus`, like `--gpus all` for example. Below is
an example for CPU-only inference:

```bash
docker run \
    -p 3000:3000 \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$HOME/models:/models" \
    llamacpp-backend \
    --model-id "Qwen/Qwen2.5-3B-Instruct" \
    --model-gguf "/models/qwen2.5-3b-instruct-q4_0.gguf"
```

This will start the server and expose the API on port 3000.

## Configuration options

The llamacpp backend provides various options to optimize performance:

| Argument                              | Description                                                            |
|---------------------------------------|------------------------------------------------------------------------|
| `--n-threads N`                       | Number of threads to use for generation                                |
| `--n-threads-batch N`                 | Number of threads to use for batch processing                          |
| `--n-gpu-layers N`                    | Number of layers to store in VRAM                                      |
| `--split-mode MODE`                   | Split the model across multiple GPUs                                   |
| `--defrag-threshold FLOAT`            | Defragment the KV cache if holes/size > threshold                      |
| `--numa MODE`                         | Enable NUMA optimizations                                              |
| `--use-mmap`                          | Use memory mapping for the model                                       |
| `--use-mlock`                         | Use memory locking to prevent swapping                                 |
| `--offload-kqv`                       | Enable offloading of KQV operations to the GPU                         |
| `--flash-attention`                   | Enable flash attention for faster inference. (EXPERIMENTAL)            |
| `--type-k TYPE`                       | Data type used for K cache                                             |
| `--type-v TYPE`                       | Data type used for V cache                                             |
| `--validation-workers N`              | Number of tokenizer workers used for payload validation and truncation |
| `--max-concurrent-requests N`         | Maximum amount of concurrent requests                                  |
| `--max-input-tokens N`                | Maximum number of input tokens per request                             |
| `--max-total-tokens N`                | Maximum total tokens (input + output) per request                      |
| `--max-batch-total-tokens N`          | Maximum number of tokens in a batch                                    |
| `--max-physical-batch-total-tokens N` | Maximum number of tokens in a physical batch                           |
| `--max-batch-size N`                  | Maximum number of requests per batch                                   |

You can also run the docker with `--help` for more information.
