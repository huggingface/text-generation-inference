ARG CUDA_ARCH_LIST="75-real;80-real;86-real;89-real;90-real"

# Build dependencies resolver stage
FROM lukemathwalker/cargo-chef:latest as chef
WORKDIR /usr/src/text-generation-inference

FROM chef as planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# CUDA dependent dependencies resolver stage
FROM nvcr.io/nvidia/pytorch:24.05-py3 as cuda-builder

RUN apt update && apt install -y \
    cmake \
    gcc  \
    g++ \
    git \
    git-lfs \
    ninja-build

# Install TensorRT
COPY backends/trtllm/scripts/install_tensorrt.sh /opt/install_tensorrt.sh
RUN chmod +x /opt/install_tensorrt.sh && \
    /opt/install_tensorrt.sh

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y && \
    chmod -R a+w $HOME/.rustup && \
    chmod -R a+w $HOME/.cargo

ENV PATH="$HOME/.cargo/bin:$PATH"
RUN $HOME/.cargo/bin/cargo install cargo-chef

# Backend build step
WORKDIR /usr/src/text-generation-inference

# Cache dependencies
COPY --from=planner /usr/src/text-generation-inference/recipe.json .
RUN $HOME/.cargo/bin/cargo chef cook --release --recipe-path recipe.json

# Build actual TGI
COPY . .
RUN $HOME/.cargo/bin/cargo build --release --bin text-generation-backends-trtllm

FROM nvcr.io/nvidia/pytorch:24.05-py3
WORKDIR /opt
COPY --from=cuda-builder /usr/local/tensorrt /usr/local/tensorrt
COPY --from=cuda-builder /usr/src/text-generation-inference/target/release/text-generation-backends-trtllm /opt/text-generation-launcher