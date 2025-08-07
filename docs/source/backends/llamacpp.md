# Llamacpp Backend

The llamacpp backend facilitates the deployment of large language models
(LLMs) by integrating [llama.cpp][llama.cpp], an advanced inference engine
optimized for both CPU and GPU computation. This backend is a component
of Hugging Face’s **Text Generation Inference (TGI)** suite,
specifically designed to streamline the deployment of LLMs in production
environments.

## Key Capabilities

- Full compatibility with GGUF format and all quantization formats
  (GGUF-related constraints may be mitigated dynamically by on-the-fly
  generation in future updates)
- Optimized inference on CPU and GPU architectures with two dedicated Dockerfiles
- Containerized deployment, eliminating dependency complexity
- Seamless interoperability with the Hugging Face ecosystem

## Model Compatibility

This backend leverages models formatted in **GGUF**, providing an
optimized balance between computational efficiency and model accuracy.
You will find the best models on [Hugging Face][GGUF].

## Build Docker image

### For CPU-only inference

For optimal performance, the Docker image is compiled with native CPU
instructions by default. As a result, it is strongly recommended to run
the container on the same host architecture used during the build
process. Efforts are ongoing to improve portability across different
systems while preserving high computational efficiency.

To build the Docker image fo CPU, use the following command:

```bash
docker build \
    -t tgi-llamacpp-cpu \
    https://github.com/huggingface/text-generation-inference.git \
    -f Dockerfile_llamacpp_cpuonly
```

### For GPU-enabled inference

```bash
docker build \
    -t tgi-llamacpp \
    https://github.com/huggingface/text-generation-inference.git \
    -f Dockerfile_llamacpp
```

### Build parameters

| Parameter (with --build-arg)              | Description                      | CPU or GPU? |
| ----------------------------------------- | -------------------------------- | ----------- |
| `llamacpp_version=bXXXX`                  | Specific version of llama.cpp    | Both        |
| `llamacpp_cuda=ON`                        | Enables CUDA acceleration        | GPU         |
| `llamacpp_native=OFF`                     | Disable automatic CPU detection  | Both        |
| `llamacpp_cpu_arm_arch=ARCH[+FEATURE]...` | Specific ARM CPU and features    | Both        |
| `cuda_arch=ARCH`                          | Defines target CUDA architecture | GPU         |

For example, to target Graviton4 when building on another ARM
architecture:

```bash
docker build \
    -t tgi-llamacpp \
    --build-arg llamacpp_native=OFF \
    --build-arg llamacpp_cpu_arm_arch=armv9-a+i8mm \
    https://github.com/huggingface/text-generation-inference.git \
    -f Dockerfile_llamacpp
```

For example, to target a local CPU without GPU acceleration:

```bash
docker build \                            
    -t tgi-llamacpp-cpu \
    --build-arg llamacpp_native=ON \
    https://github.com/huggingface/text-generation-inference.git \
    -f Dockerfile_llamacpp_cpuonly
```

As a rule of thumb, if you are not interested in GPU acceleration,
you should build the CPU-only image, which is significantly smaller 
(Dockerfile_llamacpp_cpuonly - 1.7GB vs Dockerfile_llamacpp - 17GB)

## Run Docker image

### CPU-based inference

```bash
docker run \
    -p 3000:3000 \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$HOME/models:/app/models" \
    tgi-llamacpp-cpu \
    --model-id "Qwen/Qwen2.5-3B-Instruct"
```

### GPU-Accelerated inference

```bash
docker run \
    --gpus all \
    -p 3000:3000 \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$HOME/models:/app/models" \
    tgi-llamacpp \
    --n-gpu-layers 99
    --model-id "Qwen/Qwen2.5-3B-Instruct"
```

## Using a custom GGUF

GGUF files are optional as they will be automatically generated at
startup if not already present in the `models` directory. However, if
the default GGUF generation is not suitable for your use case, you can
provide your own GGUF file with `--model-gguf`, for example:

```bash
docker run \
    -p 3000:3000 \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$HOME/models:/app/models" \
    tgi-llamacpp \
    --model-id "Qwen/Qwen2.5-3B-Instruct" \
    --model-gguf "models/qwen2.5-3b-instruct-q4_0.gguf"
```

Note that `--model-id` is still required.

## Advanced parameters

A full listing of configurable parameters is available in the `--help`:

```bash
docker run tgi-llamacpp --help

```

The table below summarizes key options:

| Parameter                           | Description                                                            |
|-------------------------------------|------------------------------------------------------------------------|
| `--n-threads`                       | Number of threads to use for generation                                |
| `--n-threads-batch`                 | Number of threads to use for batch processing                          |
| `--n-gpu-layers`                    | Number of layers to store in VRAM                                      |
| `--split-mode`                      | Split the model across multiple GPUs                                   |
| `--defrag-threshold`                | Defragment the KV cache if holes/size > threshold                      |
| `--numa`                            | Enable NUMA optimizations                                              |
| `--disable-mmap`                    | Disable memory mapping for the model                                   |
| `--use-mlock`                       | Use memory locking to prevent swapping                                 |
| `--disable-offload-kqv`             | Disable offloading of KQV operations to the GPU                        |
| `--disable-flash-attention`         | Disable flash attention                                                |
| `--type-k`                          | Data type used for K cache                                             |
| `--type-v`                          | Data type used for V cache                                             |
| `--validation-workers`              | Number of tokenizer workers used for payload validation and truncation |
| `--max-concurrent-requests`         | Maximum number of concurrent requests                                  |
| `--max-input-tokens`                | Maximum number of input tokens per request                             |
| `--max-total-tokens`                | Maximum number of total tokens (input + output) per request            |
| `--max-batch-total-tokens`          | Maximum number of tokens in a batch                                    |
| `--max-physical-batch-total-tokens` | Maximum number of tokens in a physical batch                           |
| `--max-batch-size`                  | Maximum number of requests per batch                                   |

---
[llama.cpp]: https://github.com/ggerganov/llama.cpp
[GGUF]: https://huggingface.co/models?library=gguf&sort=trending
