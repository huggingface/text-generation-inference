# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.79 AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef as planner
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -vOL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --profile release-opt --recipe-path recipe.json

ARG GIT_SHA
ARG DOCKER_LABEL

COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY launcher launcher
RUN cargo build --profile release-opt

# Python builder
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as pytorch-install

ARG PYTORCH_VERSION=2.3.0
ARG PYTHON_VERSION=3.10
# Keep in sync with `server/pyproject.toml
ARG CUDA_VERSION=12.1
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
FROM pytorch-install as kernel-builder

ARG MAX_JOBS=8

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ninja-build cmake \
        && rm -rf /var/lib/apt/lists/*

# Build Flash Attention CUDA kernels
FROM kernel-builder as flash-att-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att Makefile

# Build specific version of flash attention
RUN make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM kernel-builder as flash-att-v2-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att-v2 Makefile

# Build specific version of flash attention v2
RUN make build-flash-attention-v2-cuda

# Build Transformers exllama kernels
FROM kernel-builder as exllama-kernels-builder
WORKDIR /usr/src
COPY server/exllama_kernels/ .

RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build Transformers exllama kernels
FROM kernel-builder as exllamav2-kernels-builder
WORKDIR /usr/src
COPY server/exllamav2_kernels/ .

# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build Transformers awq kernels
FROM kernel-builder as awq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-awq Makefile
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" make build-awq

# Build eetq kernels
FROM kernel-builder as eetq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-eetq Makefile
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" make build-eetq

# Build marlin kernels
FROM kernel-builder as marlin-kernels-builder
WORKDIR /usr/src
COPY server/marlin/ .
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build Transformers CUDA kernels
FROM kernel-builder as custom-kernels-builder
WORKDIR /usr/src
COPY server/custom_kernels/ .
# Build specific version of transformers
RUN python setup.py build

# Build vllm CUDA kernels
FROM kernel-builder as vllm-builder

WORKDIR /usr/src

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

COPY server/Makefile-vllm Makefile

# Build specific version of vllm
RUN make build-vllm-cuda

# Build mamba kernels
FROM kernel-builder as mamba-builder
WORKDIR /usr/src
COPY server/Makefile-selective-scan Makefile
RUN make build-all

# Text Generation Inference base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 as base

# Conda env
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# Text Generation Inference base env
ENV HUGGINGFACE_HUB_CACHE=/data \
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
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so /opt/conda/lib/python3.10/site-packages

# Copy build artifacts from custom kernels builder
COPY --from=custom-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
# Copy build artifacts from exllamav2 kernels builder
COPY --from=exllamav2-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
# Copy build artifacts from awq kernels builder
COPY --from=awq-kernels-builder /usr/src/llm-awq/awq/kernels/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
# Copy build artifacts from eetq kernels builder
COPY --from=eetq-kernels-builder /usr/src/eetq/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
# Copy build artifacts from marlin kernels builder
COPY --from=marlin-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy builds artifacts from vllm builder
COPY --from=vllm-builder /usr/src/vllm/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy build artifacts from mamba builder
COPY --from=mamba-builder /usr/src/mamba/build/lib.linux-x86_64-cpython-310/ /opt/conda/lib/python3.10/site-packages
COPY --from=mamba-builder /usr/src/causal-conv1d/build/lib.linux-x86_64-cpython-310/ /opt/conda/lib/python3.10/site-packages

# Install flash-attention dependencies
RUN pip install einops --no-cache-dir

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile
RUN cd server && \
    make gen-server && \
    pip install -r requirements_cuda.txt && \
    pip install ".[bnb, accelerate, quantize, peft, outlines]" --no-cache-dir

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
FROM base as sagemaker

COPY sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM base

COPY ./tgi-entrypoint.sh /tgi-entrypoint.sh
RUN chmod +x /tgi-entrypoint.sh

ENTRYPOINT ["/tgi-entrypoint.sh"]
CMD ["--json-output"]
