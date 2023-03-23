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

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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
    LD_LIBRARY_PATH="/opt/miniconda/envs/text-generation/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
    CONDA_DEFAULT_ENV=text-generation \
    PATH=$PATH:/opt/miniconda/envs/text-generation/bin:/opt/miniconda/bin:/usr/local/cuda/bin

RUN apt-get update && apt-get install -y git curl libssl-dev && rm -rf /var/lib/apt/lists/*

RUN cd ~ && \
    curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -bf -p /opt/miniconda && \
    conda create -n text-generation python=3.9 -y

WORKDIR /usr/src

# Install torch
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

COPY server/Makefile server/Makefile

# Install specific version of flash attention
RUN cd server && make install-flash-attention

# Install specific version of transformers
RUN cd server && BUILD_EXTENSIONS="True" make install-transformers

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    /opt/miniconda/envs/text-generation/bin/pip install ".[bnb]" --no-cache-dir

# Install router
COPY --from=builder /usr/src/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]