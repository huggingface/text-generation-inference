# TensorRT-LLM backend

The NVIDIA TensorRT-LLM (TRTLLM) backend is a high-performance backend for LLMs
that uses NVIDIA's TensorRT library for inference acceleration.
It makes use of specific optimizations for NVIDIA GPUs, such as custom kernels.

To use the TRTLLM backend **you need to compile** `engines` for the models you want to use.
Each `engine` must be compiled for a given set of:
- GPU architecture that you will use for inference (e.g. A100, L40, etc.)
- Maximum batch size
- Maximum input length
- Maximum output length
- Maximum beams width

## Supported models

Check the [support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) to see which models are
supported.

## Compiling engines

You can use [Optimum-NVIDIA](https://github.com/huggingface/optimum-nvidia) to compile engines for the models you
want to use.

```bash
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DESTINATION="/tmp/engines/$MODEL_NAME"
HF_TOKEN="hf_xxx"
# Compile the engine using Optimum-NVIDIA
# This will create a compiled engine in the /tmp/engines/meta-llama/Llama-3.1-8B-Instruct
# directory for 1 GPU
docker run \
  --rm \
  -it \
  --gpus=1 \
  --shm-size=1g \
  -v "$DESTINATION":/engine \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  huggingface/optimum-nvidia:v0.1.0b9-py310 \
    bash -c "optimum-cli export trtllm \
    --tp=1 \
    --pp=1 \
    --max-batch-size=64 \
    --max-input-length 4096 \
    --max-output-length 8192 \
    --max-beams-width=1 \
    --destination /tmp/engine \
    $MODEL_NAME && cp -rL /tmp/engine/* /engine/"
```

Your compiled engine will be saved in the `/tmp/engines/$MODEL_NAME` directory, in a subfolder named after the GPU used to compile the model.

## Using the TRTLLM backend

Run TGI-TRTLLM Docker image with the compiled engine:

```bash
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DESTINATION="/tmp/engines/$MODEL_NAME"
HF_TOKEN="hf_xxx"
docker run \
  --gpus 1 \
  --shm-size=1g \
  -it \
  --rm \
  -p 3000:3000 \
  -e MODEL=$MODEL_NAME \
  -e PORT=3000 \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$DESTINATION"/<YOUR_GPU_ARCHITECTURE>/engines:/data \
  ghcr.io/huggingface/text-generation-inference:latest-trtllm \
  --model-id /data/ \
  --tokenizer-name $MODEL_NAME
```

## Development

To develop TRTLLM backend, you can use [dev containers](https://containers.dev/) with the following `.devcontainer.json` file:
```json
{
  "name": "CUDA",
  "build": {
    "dockerfile": "Dockerfile_trtllm",
    "context": ".."
  },
  "remoteEnv": {
    "PATH": "${containerEnv:PATH}:/usr/local/cuda/bin",
    "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64",
    "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda"
  },
  "customizations" : {
    "jetbrains" : {
      "backend" : "CLion"
    }
  }
}
```

and `Dockerfile_trtllm`:

```Dockerfile
ARG cuda_arch_list="75-real;80-real;86-real;89-real;90-real"
ARG build_type=release
ARG ompi_version=4.1.7

# CUDA dependent dependencies resolver stage
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS cuda-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    curl \
    gcc-14  \
    g++-14 \
    git \
    git-lfs \
    lld \
    libssl-dev \
    libucx-dev \
    libasan8 \
    libubsan1 \
    ninja-build \
    pkg-config \
    pipx \
    python3 \
    python3-dev \
    python3-setuptools \
    tar \
    wget --no-install-recommends && \
    pipx ensurepath

ENV TGI_INSTALL_PREFIX=/usr/local/tgi
ENV TENSORRT_INSTALL_PREFIX=/usr/local/tensorrt

# Install OpenMPI
FROM cuda-builder AS mpi-builder
WORKDIR /opt/src/mpi

ARG ompi_version
ENV OMPI_VERSION=${ompi_version}
ENV OMPI_TARBALL_FILENAME=openmpi-${OMPI_VERSION}.tar.bz2
ADD --checksum=sha256:54a33cb7ad81ff0976f15a6cc8003c3922f0f3d8ceed14e1813ef3603f22cd34 \
    https://download.open-mpi.org/release/open-mpi/v4.1/${OMPI_TARBALL_FILENAME} .

RUN tar --strip-components=1 -xf ${OMPI_TARBALL_FILENAME} &&\
    ./configure --prefix=/usr/local/mpi --with-cuda=/usr/local/cuda --with-slurm && \
    make -j all && \
    make install && \
    rm -rf ${OMPI_TARBALL_FILENAME}/..

# Install TensorRT
FROM cuda-builder AS trt-builder
COPY backends/trtllm/scripts/install_tensorrt.sh /opt/install_tensorrt.sh
RUN chmod +x /opt/install_tensorrt.sh && \
    /opt/install_tensorrt.sh

# Build Backend
FROM cuda-builder AS tgi-builder
WORKDIR /usr/src/text-generation-inference

# Scoped global args reuse
ARG cuda_arch_list
ARG build_type
ARG sccache_gha_enabled
ARG actions_cache_url
ARG actions_runtime_token

# Install Rust
ENV PATH="/root/.cargo/bin:$PATH"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y && \
    chmod -R a+w /root/.rustup && \
    chmod -R a+w /root/.cargo && \
    cargo install sccache --locked

ENV LD_LIBRARY_PATH="/usr/local/mpi/lib:$LD_LIBRARY_PATH"
ENV PKG_CONFIG_PATH="/usr/local/mpi/lib/pkgconfig"
ENV CMAKE_PREFIX_PATH="/usr/local/mpi:/usr/local/tensorrt"

ENV USE_LLD_LINKER=ON
ENV CUDA_ARCH_LIST=${cuda_arch_list}
```
