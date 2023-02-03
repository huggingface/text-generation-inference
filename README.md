<div align="center">

# Text Generation Inference

<a href="https://github.com/huggingface/text-generation-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-generation-inference?style=social">
</a>
<a href="https://github.com/huggingface/text-generation-inference/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/github/license/huggingface/text-generation-inference">
</a>
<a href="https://huggingface.github.io/text-generation-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

![architecture](assets/architecture.jpg)

</div>

A Rust, Python and gRPC server for text generation inference. Used in production at [HuggingFace](https://huggingface.co) 
to power LLMs api-inference widgets.

## Table of contents

- [Features](#features)
- [Officially Supported Models](#officially-supported-models)
- [Get Started](#get-started)
  - [Docker](#docker)
  - [Local Install](#local-install)
  - [OpenAPI](#api-documentation)
  - [CUDA Kernels](#cuda-kernels)
- [Run BLOOM](#run-bloom)
  - [Download](#download)
  - [Run](#run)
  - [Quantization](#quantization)
- [Develop](#develop)
- [Testing](#testing)
  
## Features

- Token streaming using Server Side Events (SSE)
- [Dynamic batching of incoming requests](https://github.com/huggingface/text-generation-inference/blob/main/router/src/batcher.rs#L88) for increased total throughput
- Quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Safetensors](https://github.com/huggingface/safetensors) weight loading
- 45ms per token generation for BLOOM with 8xA100 80GB
- Logits warpers (temperature scaling, topk, repetition penalty ...)
- Stop sequences
- Log probabilities

## Officially supported models

- [BLOOM](https://huggingface.co/bigscience/bloom)
- [BLOOMZ](https://huggingface.co/bigscience/bloomz)
- [MT0-XXL](https://huggingface.co/bigscience/mt0-xxl)
- ~~[Galactica](https://huggingface.co/facebook/galactica-120b)~~ (deactivated)
- [SantaCoder](https://huggingface.co/bigcode/santacoder)
- [GPT-Neox 20B](https://huggingface.co/EleutherAI/gpt-neox-20b): use `--revision pr/13`

Other models are supported on a best effort basis using:

`AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")`

or

`AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")`

## Get started

### Docker

The easiest way of getting started is using the official Docker container:

```shell
model=bigscience/bloom-560m
num_shard=2
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --num-shard $num_shard
```

You can then query the model using either the `/generate` or `/generate_stream` routes:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Testing API","parameters":{"max_new_tokens":9}}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"Testing API","parameters":{"max_new_tokens":9}}' \
    -H 'Content-Type: application/json'
```

**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### API documentation

You can consult the OpenAPI documentation of the `text-generation-inference` REST API using the `/docs` route.
The Swagger UI is also available at: [https://huggingface.github.io/text-generation-inference](https://huggingface.github.io/text-generation-inference).

### Local install

You can also opt to install `text-generation-inference` locally. 

First [install Rust](https://rustup.rs/) and create a Python virtual environment with at least 
Python 3.9, e.g. using `conda`:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

conda create -n text-generation-inference python=3.9 
conda activate text-generation-inference
```

Then run:

```shell
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels
make run-bloom-560m
```

**Note:** on some machines, you may also need the OpenSSL libraries. On Linux machines, run:

```shell
sudo apt-get install libssl-dev
```

### CUDA Kernels

The custom CUDA kernels are only tested on NVIDIA A100s. If you have any installation or runtime issues, you can remove 
the kernels by using the `BUILD_EXTENSIONS=False` environment variable.

Be aware that the official Docker image has them enabled by default.

## Run BLOOM

### Download

First you need to download the weights:

```shell
make download-bloom
```

### Run

```shell
make run-bloom # Requires 8xA100 80GB
```

### Quantization

You can also quantize the weights with bitsandbytes to reduce the VRAM requirement:

```shell
make run-bloom-quantize # Requires 8xA100 40GB
```

## Develop

```shell
make server-dev
make router-dev
```

## Testing

```shell
make python-tests
make integration-tests
```
