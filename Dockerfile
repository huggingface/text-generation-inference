FROM rust:1.64 as builder

WORKDIR /usr/src

COPY proto proto
COPY router router

WORKDIR /usr/src/router

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

RUN apt-get update && apt-get install -y unzip wget libssl-dev && rm -rf /var/lib/apt/lists/*

RUN cd ~ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -bf -p /opt/miniconda && \
    conda create -n text-generation python=3.9 -y

# Install specific version of torch
RUN /opt/miniconda/envs/text-generation/bin/pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir

# Install specific version of transformers
RUN wget https://github.com/huggingface/transformers/archive/46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip && \
    unzip 46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip && \
    rm 46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip && \
    cd transformers-46d37bece7d3ffdef97b1ee4a3170c0a0627d921 && \
    /opt/miniconda/envs/text-generation/bin/python setup.py install

WORKDIR /usr/src

# Install server
COPY server server
RUN cd server && \
    /opt/miniconda/envs/text-generation/bin/pip install . --no-cache-dir

# Install router
COPY --from=builder /usr/local/cargo/bin/text-generation-router /usr/local/bin/text-generation-router

COPY run.sh .
RUN chmod +x run.sh

CMD ["./run.sh"]