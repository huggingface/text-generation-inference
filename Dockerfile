# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.80.1 AS chef
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

# NOTE: When updating PyTorch version, beware to remove `pip install nvidia-nccl-cu12==2.22.3` below in the Dockerfile. Context: https://github.com/huggingface/text-generation-inference/pull/2099
ARG PYTORCH_VERSION=2.4.0

ARG PYTHON_VERSION=3.11
# Keep in sync with `server/pyproject.toml
ARG CUDA_VERSION=12.4
ARG MAMBA_VERSION=24.3.0-0
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*

# Install conda
# translating Docker's TARGETPLATFORM into mamba arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
         *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/mambaforge.sh -O  "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh"
RUN chmod +x ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

# Install pytorch
# On arm64 we exit with an error code
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  exit 1 ;; \
         *)              /opt/conda/bin/conda update -y conda &&  \
                         /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" "pytorch=$PYTORCH_VERSION" "pytorch-cuda=$(echo $CUDA_VERSION | cut -d'.' -f 1-2)"  ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

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
RUN make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM kernel-builder AS flash-att-v2-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att-v2 Makefile

# Build specific version of flash attention v2
RUN make build-flash-attention-v2-cuda

# Build Transformers exllama kernels
FROM kernel-builder AS exllama-kernels-builder
WORKDIR /usr/src
COPY server/exllama_kernels/ .

RUN python setup.py build

# Build Transformers exllama kernels
FROM kernel-builder AS exllamav2-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-exllamav2/ Makefile

# Build specific version of transformers
RUN make build-exllamav2

# Build Transformers awq kernels
FROM kernel-builder AS awq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-awq Makefile
# Build specific version of transformers
RUN make build-awq

# Build eetq kernels
FROM kernel-builder AS eetq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-eetq Makefile
# Build specific version of transformers
RUN make build-eetq

# Build Lorax Punica kernels
FROM kernel-builder AS lorax-punica-builder
WORKDIR /usr/src
COPY server/Makefile-lorax-punica Makefile
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" make build-lorax-punica

# Build Transformers CUDA kernels
FROM kernel-builder AS custom-kernels-builder
WORKDIR /usr/src
COPY server/custom_kernels/ .
# Build specific version of transformers
RUN python setup.py build

# Build mamba kernels
FROM kernel-builder AS mamba-builder
WORKDIR /usr/src
COPY server/Makefile-selective-scan Makefile
RUN make build-all

# Build flashinfer
FROM kernel-builder AS flashinfer-builder
WORKDIR /usr/src
COPY server/Makefile-flashinfer Makefile
RUN make install-flashinfer

# Text Generation Inference base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS base

# Conda env
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

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

# Copy conda with PyTorch installed
COPY --from=pytorch-install /opt/conda /opt/conda

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /opt/conda/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so /opt/conda/lib/python3.11/site-packages

# Copy build artifacts from custom kernels builder
COPY --from=custom-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from exllamav2 kernels builder
COPY --from=exllamav2-kernels-builder /usr/src/exllamav2/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from awq kernels builder
COPY --from=awq-kernels-builder /usr/src/llm-awq/awq/kernels/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from eetq kernels builder
COPY --from=eetq-kernels-builder /usr/src/eetq/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from lorax punica kernels builder
COPY --from=lorax-punica-builder /usr/src/lorax-punica/server/punica_kernels/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from mamba builder
COPY --from=mamba-builder /usr/src/mamba/build/lib.linux-x86_64-cpython-311/ /opt/conda/lib/python3.11/site-packages
COPY --from=mamba-builder /usr/src/causal-conv1d/build/lib.linux-x86_64-cpython-311/ /opt/conda/lib/python3.11/site-packages
COPY --from=flashinfer-builder /opt/conda/lib/python3.11/site-packages/flashinfer/ /opt/conda/lib/python3.11/site-packages/flashinfer/

# Install flash-attention dependencies
RUN pip install einops --no-cache-dir

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile
RUN cd server && \
    make gen-server && \
    pip install -r requirements_cuda.txt && \
    pip install ".[attention, bnb, accelerate, compressed-tensors, marlin, moe, quantize, peft, outlines]" --no-cache-dir && \
    pip install nvidia-nccl-cu12==2.22.3

ENV LD_PRELOAD=/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2
# Required to find libpython within the rust binaries
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib/"
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

ENTRYPOINT ["/tgi-entrypoint.sh"]
# CMD ["--json-output"]
