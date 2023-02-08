FROM rust:1.65 as router-builder

WORKDIR /usr/src

COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY router router

WORKDIR /usr/src/router

RUN cargo install --path .

FROM rust:1.65 as launcher-builder

WORKDIR /usr/src

COPY rust-toolchain.toml rust-toolchain.toml
COPY launcher launcher

WORKDIR /usr/src/launcher

RUN cargo install --path .

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    HUGGINGFACE_HUB_CACHE=/data \
    MODEL_ID=bigscience/bloom-560m \
    QUANTIZE=false \
    NUM_SHARD=1 \
    PORT=80 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH="/opt/miniconda/envs/text-generation/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
    CONDA_DEFAULT_ENV=text-generation \
    PATH=$PATH:/opt/miniconda/envs/text-generation/bin:/opt/miniconda/bin:/usr/local/cuda/bin

RUN apt-get update && apt-get install -y unzip curl libssl-dev && rm -rf /var/lib/apt/lists/*

RUN cd ~ && \
    curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -bf -p /opt/miniconda && \
    conda create -n text-generation python=3.9 -y

WORKDIR /usr/src

COPY server/Makefile server/Makefile

# Install specific version of torch
RUN cd server && make install-torch

# Install specific version of transformers
RUN cd server && BUILD_EXTENSIONS="True" make install-transformers

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    /opt/miniconda/envs/text-generation/bin/pip install ".[bnb]" --no-cache-dir

# Install router
COPY --from=router-builder /usr/local/cargo/bin/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=launcher-builder /usr/local/cargo/bin/text-generation-launcher /usr/local/bin/text-generation-launcher

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]