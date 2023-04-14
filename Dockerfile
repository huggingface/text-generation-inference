FROM lukemathwalker/cargo-chef:latest-rust-1.67 AS chef
WORKDIR /usr/src

FROM chef as planner
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY router router
COPY launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY router router
COPY launcher launcher
RUN cargo build --release

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

ARG MAMBA_VERSION=23.1.0-1

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=bigscience/bloom-560m \
    QUANTIZE=false \
    NUM_SHARD=1 \
    PORT=80 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH="/opt/conda/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
    PATH=$PATH:/opt/conda/bin:/usr/local/cuda/bin

RUN apt-get update && apt-get install -y git curl libssl-dev ninja-build && rm -rf /var/lib/apt/lists/*

RUN cd ~ && \
    curl -fsSL -v -o ~/mambaforge.sh -O  "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-x86_64.sh" \
    chmod +x ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

WORKDIR /usr/src

# Install torch
RUN pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Install specific version of flash attention
COPY server/Makefile-flash-att server/Makefile
RUN cd server && make install-flash-attention

# Install specific version of transformers
COPY server/Makefile-transformers server/Makefile
RUN cd server && BUILD_EXTENSIONS="True" make install-transformers

COPY server/Makefile server/Makefile

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    pip install ".[bnb]" --no-cache-dir

# Install router
COPY --from=builder /usr/src/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher

# AWS Sagemaker compatbile image
FROM base as sagemaker

COPY sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM base

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]