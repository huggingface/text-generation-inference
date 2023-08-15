<div align="center">

![image](https://github.com/huggingface/text-generation-inference/assets/3841370/38ba1531-ea0d-4851-b31a-a6d4ddc944b0)

# Text Generation Inference

<a href="https://github.com/huggingface/text-generation-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-generation-inference?style=social">
</a>
<a href="https://huggingface.github.io/text-generation-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A Rust, Python and gRPC server for text generation inference. Used in production at [HuggingFace](https://huggingface.co)
to power Hugging Chat, the Inference API and Inference Endpoint.

</div>

## Table of contents

- [Features](#features)
- [Optimized Architectures](#optimized-architectures)
- [Get Started](#get-started)
  - [Docker](#docker)
  - [API Documentation](#api-documentation)
  - [Using a private or gated model](#using-a-private-or-gated-model)
  - [A note on Shared Memory](#a-note-on-shared-memory-shm)
  - [Distributed Tracing](#distributed-tracing)
  - [Local Install](#local-install)
  - [CUDA Kernels](#cuda-kernels)
- [Run Falcon](#run-falcon)
  - [Run](#run)
  - [Quantization](#quantization)
- [Develop](#develop)
- [Testing](#testing)
- [Other supported hardware](#other-supported-hardware)

## Features

- Serve the most popular Large Language Models with a simple launcher
- Tensor Parallelism for faster inference on multiple GPUs
- Token streaming using Server-Sent Events (SSE)
- [Continuous batching of incoming requests](https://github.com/huggingface/text-generation-inference/tree/main/router) for increased total throughput
- Optimized transformers code for inference using [flash-attention](https://github.com/HazyResearch/flash-attention) and [Paged Attention](https://github.com/vllm-project/vllm) on the most popular architectures
- Quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPT-Q](https://arxiv.org/abs/2210.17323)
- [Safetensors](https://github.com/huggingface/safetensors) weight loading
- Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
- Logits warper (temperature scaling, top-p, top-k, repetition penalty, more details see [transformers.LogitsProcessor](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.LogitsProcessor))
- Stop sequences
- Log probabilities
- Production ready (distributed tracing with Open Telemetry, Prometheus metrics)
- Custom Prompt Generation: Easily generate text by providing custom prompts to guide the model's output.
- Fine-tuning Support: Utilize fine-tuned models for specific tasks to achieve higher accuracy and performance.


## Optimized architectures

- [BLOOM](https://huggingface.co/bigscience/bloom)
- [FLAN-T5](https://huggingface.co/google/flan-t5-xxl)
- [Galactica](https://huggingface.co/facebook/galactica-120b)
- [GPT-Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [Llama](https://github.com/facebookresearch/llama)
- [OPT](https://huggingface.co/facebook/opt-66b)
- [SantaCoder](https://huggingface.co/bigcode/santacoder)
- [Starcoder](https://huggingface.co/bigcode/starcoder)
- [Falcon 7B](https://huggingface.co/tiiuae/falcon-7b)
- [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b)
- [MPT](https://huggingface.co/mosaicml/mpt-30b)
- [Llama V2](https://huggingface.co/meta-llama)

Other architectures are supported on a best effort basis using:

`AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")`

or

`AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")`

## Get started

### Docker

The easiest way of getting started is using the official Docker container:

```shell
model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.1 --model-id $model
```
**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher. For running the Docker container on a machine with no GPUs or CUDA support, it is enough to remove the `--gpus all` flag and add `--disable-custom-kernels`, please note CPU is not the intended platform for this project, so performance might be subpar.

To see all options to serve your models (in the [code](https://github.com/huggingface/text-generation-inference/blob/main/launcher/src/main.rs) or in the cli):
```
text-generation-launcher --help
```

You can then query the model using either the `/generate` or `/generate_stream` routes:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

or from Python:

```shell
pip install text-generation
```

```python
from text_generation import Client

client = Client("http://127.0.0.1:8080")
print(client.generate("What is Deep Learning?", max_new_tokens=20).generated_text)

text = ""
for response in client.generate_stream("What is Deep Learning?", max_new_tokens=20):
    if not response.token.special:
        text += response.token.text
print(text)
```

### API documentation

You can consult the OpenAPI documentation of the `text-generation-inference` REST API using the `/docs` route.
The Swagger UI is also available at: [https://huggingface.github.io/text-generation-inference](https://huggingface.github.io/text-generation-inference).

### Using a private or gated model

You have the option to utilize the `HUGGING_FACE_HUB_TOKEN` environment variable for configuring the token employed by
`text-generation-inference`. This allows you to gain access to protected resources.

For example, if you want to serve the gated Llama V2 model variants:

1. Go to https://huggingface.co/settings/tokens
2. Copy your cli READ token
3. Export `HUGGING_FACE_HUB_TOKEN=<your cli READ token>`

or with Docker:

```shell
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your cli READ token>

docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.1 --model-id $model
```

### A note on Shared Memory (shm)

[`NCCL`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) is a communication framework used by
`PyTorch` to do distributed training/inference. `text-generation-inference` make
use of `NCCL` to enable Tensor Parallelism to dramatically speed up inference for large language models.

In order to share data between the different devices of a `NCCL` group, `NCCL` might fall back to using the host memory if
peer-to-peer using NVLink or PCI is not possible.

To allow the container to use 1G of Shared Memory and support SHM sharing, we add `--shm-size 1g` on the above command.

If you are running `text-generation-inference` inside `Kubernetes`. You can also add Shared Memory to the container by
creating a volume with:

```yaml
- name: shm
  emptyDir:
   medium: Memory
   sizeLimit: 1Gi
```

and mounting it to `/dev/shm`.

Finally, you can also disable SHM sharing by using the `NCCL_SHM_DISABLE=1` environment variable. However, note that
this will impact performance.

### Distributed Tracing

`text-generation-inference` is instrumented with distributed tracing using OpenTelemetry. You can use this feature
by setting the address to an OTLP collector with the `--otlp-endpoint` argument.

### Local install

You can also opt to install `text-generation-inference` locally.

First [install Rust](https://rustup.rs/) and create a Python virtual environment with at least
Python 3.9, e.g. using `conda`:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

conda create -n text-generation-inference python=3.9
conda activate text-generation-inference
```

You may also need to install Protoc.

On Linux:

```shell
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

On MacOS, using Homebrew:

```shell
brew install protobuf
```

Then run:

```shell
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels
make run-falcon-7b-instruct
```

**Note:** on some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```shell
sudo apt-get install libssl-dev gcc -y
```

### CUDA Kernels

The custom CUDA kernels are only tested on NVIDIA A100s. If you have any installation or runtime issues, you can remove
the kernels by using the `DISABLE_CUSTOM_KERNELS=True` environment variable.

Be aware that the official Docker image has them enabled by default.

## Run Falcon

### Run

```shell
make run-falcon-7b-instruct
```

### Quantization

You can also quantize the weights with bitsandbytes to reduce the VRAM requirement:

```shell
make run-falcon-7b-instruct-quantize
```

4bit quantization is available using the [NF4 and FP4 data types from bitsandbytes](https://arxiv.org/pdf/2305.14314.pdf). It can be enabled by providing `--quantize bitsandbytes-nf4` or `--quantize bitsandbytes-fp4` as a command line argument to `text-generation-launcher`.

## Develop

```shell
make server-dev
make router-dev
```

## Testing

```shell
# python
make python-server-tests
make python-client-tests
# or both server and client tests
make python-tests
# rust cargo tests
make rust-tests
# integration tests
make integration-tests
```


## Other supported hardware

TGI is also supported on the following AI hardware accelerators:
- *Habana first-gen Gaudi and Gaudi2:* checkout [here](https://github.com/huggingface/optimum-habana/tree/main/text-generation-inference) how to serve models with TGI on Gaudi and Gaudi2 with [Optimum Habana](https://huggingface.co/docs/optimum/habana/index)

