# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.85.1 AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef AS planner
COPY Cargo.lock Cargo.lock
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY backends backends
COPY launcher launcher

RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.11-dev
RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --profile release-opt --recipe-path recipe.json

ARG GIT_SHA
ARG DOCKER_LABEL

COPY Cargo.lock Cargo.lock
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY backends backends
COPY launcher launcher
RUN cargo build --profile release-opt --frozen

# Python builder
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS pytorch-install
WORKDIR /usr/src/

# NOTE: When updating PyTorch version, beware to remove `pip install nvidia-nccl-cu12==2.22.3` below in the Dockerfile. Context: https://github.com/huggingface/text-generation-inference/pull/2099
ARG PYTORCH_VERSION=2.6
ARG PYTHON_VERSION=3.11

# Keep in sync with `server/pyproject.toml
# Automatically set by buildx
ARG TARGETPLATFORM

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /uvx /bin/
ENV PATH="$PATH:/root/.local/bin"
RUN uv python install ${PYTHON_VERSION}
RUN uv venv --python ${PYTHON_VERSION} && uv pip install torch==${PYTORCH_VERSION} torchvision pip setuptools packaging
ENV VIRTUAL_ENV=/usr/src/.venv/
ENV PATH="$PATH:/usr/src/.venv/bin/"

# CUDA kernels builder image
FROM pytorch-install AS kernel-builder

ARG MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ninja-build cmake \
        && rm -rf /var/lib/apt/lists/*

# Build Flash Attention CUDA kernels
FROM kernel-builder AS flash-att-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att Makefile

# Build specific version of flash attention
RUN . .venv/bin/activate && make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM kernel-builder AS flash-att-v2-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att-v2 Makefile

# Build specific version of flash attention v2
RUN . .venv/bin/activate && make build-flash-attention-v2-cuda

# Build Transformers exllama kernels
FROM kernel-builder AS exllama-kernels-builder
WORKDIR /usr/src
COPY server/exllama_kernels/ .

RUN . .venv/bin/activate && python setup.py build

# Build Transformers exllama kernels
FROM kernel-builder AS exllamav2-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-exllamav2/ Makefile

# Build specific version of transformers
RUN . .venv/bin/activate && make build-exllamav2

# Build Transformers awq kernels
FROM kernel-builder AS awq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-awq Makefile
# Build specific version of transformers
RUN . .venv/bin/activate && make build-awq

# Build Lorax Punica kernels
FROM kernel-builder AS lorax-punica-builder
WORKDIR /usr/src
COPY server/Makefile-lorax-punica Makefile
# Build specific version of transformers
RUN . .venv/bin/activate && TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" make build-lorax-punica

# Build Transformers CUDA kernels
FROM kernel-builder AS custom-kernels-builder
WORKDIR /usr/src
COPY server/custom_kernels/ .
# Build specific version of transformers
RUN . .venv/bin/activate && python setup.py build

# Build mamba kernels
FROM kernel-builder AS mamba-builder
WORKDIR /usr/src
COPY server/Makefile-selective-scan Makefile
RUN . .venv/bin/activate && make build-all

# Build flashinfer
FROM kernel-builder AS flashinfer-builder
WORKDIR /usr/src
COPY server/Makefile-flashinfer Makefile
RUN . .venv/bin/activate && make install-flashinfer

# Text Generation Inference base image
FROM nvidia/cuda:12.4.0-base-ubuntu22.04 AS base

# Text Generation Inference base env
ENV HF_HOME=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

WORKDIR /usr/src

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl-dev \
        ca-certificates \
        make \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ENV PATH="$PATH:/root/.local/bin"
COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /uvx /bin/
# Install flash-attention dependencies
# RUN pip install einops --no-cache-dir

# Copy env with PyTorch installed
COPY --from=pytorch-install /usr/src/.venv /usr/src/.venv
ENV PYTHON_VERSION=3.11
RUN uv python install ${PYTHON_VERSION}
ENV VIRTUAL_ENV=/usr/src/.venv/
ENV PATH="$PATH:/usr/src/.venv/bin/"

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile
ENV HF_KERNELS_CACHE=/kernels
RUN cd server && \
	uv sync --frozen --extra gen --extra bnb --extra accelerate --extra compressed-tensors --extra quantize --extra peft --extra outlines --extra torch --no-install-project --active && \
    make gen-server-raw && \
    kernels download .

RUN cd server && \
    uv sync --frozen --extra gen --extra bnb --extra accelerate --extra compressed-tensors --extra quantize --extra peft --extra outlines --extra torch --active --python=${PYTHON_VERSION} && \
    uv pip install nvidia-nccl-cu12==2.25.1 && \
    pwd && \
    text-generation-server --help

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /usr/src/.venv/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so /usr/src/.venv/lib/python3.11/site-packages

# Copy build artifacts from custom kernels builder
COPY --from=custom-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages
# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages
# Copy build artifacts from exllamav2 kernels builder
COPY --from=exllamav2-kernels-builder /usr/src/exllamav2/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages
# Copy build artifacts from awq kernels builder
COPY --from=awq-kernels-builder /usr/src/llm-awq/awq/kernels/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages
# Copy build artifacts from lorax punica kernels builder
COPY --from=lorax-punica-builder /usr/src/lorax-punica/server/punica_kernels/build/lib.linux-x86_64-cpython-311 /usr/src/.venv/lib/python3.11/site-packages
# Copy build artifacts from mamba builder
COPY --from=mamba-builder /usr/src/mamba/build/lib.linux-x86_64-cpython-311/ /usr/src/.venv/lib/python3.11/site-packages
COPY --from=mamba-builder /usr/src/causal-conv1d/build/lib.linux-x86_64-cpython-311/ /usr/src/.venv/lib/python3.11/site-packages
COPY --from=flashinfer-builder /usr/src/.venv/lib/python3.11/site-packages/flashinfer/ /usr/src/.venv/lib/python3.11/site-packages/flashinfer/


# ENV LD_PRELOAD=/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2
# Required to find libpython within the rust binaries
# This is needed because exl2 tries to load flash-attn
# And fails with our builds.
ENV EXLLAMA_NO_FLASH_ATTN=1

# Deps before the binaries
# The binaries change on every build given we burn the SHA into them
# The deps change less often.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Install benchmarker
COPY --from=builder /usr/src/target/release-opt/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /usr/src/target/release-opt/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release-opt/text-generation-launcher /usr/local/bin/text-generation-launcher


# AWS Sagemaker compatible image
FROM base AS sagemaker

COPY sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM base

COPY ./tgi-entrypoint.sh /tgi-entrypoint.sh
RUN chmod +x /tgi-entrypoint.sh

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.local/share/uv/python/cpython-3.11.11-linux-x86_64-gnu/lib/"
ENTRYPOINT ["/tgi-entrypoint.sh"]
# CMD ["--json-output"]
