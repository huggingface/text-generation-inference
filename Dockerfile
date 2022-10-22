FROM rust:1.64 as router-builder

WORKDIR /usr/src

COPY proto proto
COPY router router

WORKDIR /usr/src/router

RUN cargo install --path .

FROM rust:1.64 as launcher-builder

WORKDIR /usr/src

COPY launcher launcher

WORKDIR /usr/src/launcher

RUN cargo install --path .

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_BASE_PATH=/var/azureml-model \
    MODEL_NAME=bigscience/bloom \
    NUM_GPUS=8 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH="/opt/miniconda/envs/text-generation/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
    CONDA_DEFAULT_ENV=text-generation \
    PATH=$PATH:/opt/miniconda/envs/text-generation/bin:/opt/miniconda/bin:/usr/local/cuda/bin

SHELL ["/bin/bash", "-c"]

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
RUN cd server && make install-transformers

# Install specific version of safetensors
# FIXME: This is a temporary fix while we wait for a new release
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cd server && make install-safetensors

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    /opt/miniconda/envs/text-generation/bin/pip install . --no-cache-dir

# Install router
COPY --from=router-builder /usr/local/cargo/bin/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=launcher-builder /usr/local/cargo/bin/text-generation-launcher /usr/local/bin/text-generation-launcher

CMD HUGGINGFACE_HUB_CACHE=$MODEL_BASE_PATH text-generation-launcher --model-name $MODEL_NAME --num-shard $NUM_GPUS