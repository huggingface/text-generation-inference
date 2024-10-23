# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.80 AS chef
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

COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY backends backends
COPY launcher launcher
RUN cargo build --profile release-opt

# Text Generation Inference base image
FROM vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest as base

# Text Generation Inference base env
ENV HF_HOME=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

# libssl.so.1.1 is not installed on Ubuntu 22.04 by default, install it
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    dpkg -i ./libssl1.1_1.1.1f-1ubuntu2_amd64.deb

WORKDIR /usr/src

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl-dev \
        ca-certificates \
        make \
        curl \
        git \
        python3.11-dev \
        && rm -rf /var/lib/apt/lists/*

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile
RUN cd server && \
    make gen-server && \
    pip install -r requirements.txt && \
    bash ./dill-0.3.8-patch.sh && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.17.0 && \
    BUILD_CUDA_EXT=0 pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@097dd04e --no-build-isolation && \
    pip install . --no-cache-dir

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
CMD ["--json-output"]
