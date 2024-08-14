ARG CUDA_ARCH_LIST="75-real;80-real;86-real;89-real;90-real"
ARG OMPI_VERSION="4.1.6"

# Build dependencies resolver stage
FROM lukemathwalker/cargo-chef:latest AS chef
WORKDIR /usr/src/text-generation-inference/backends/trtllm

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# CUDA dependent dependencies resolver stage
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04 AS cuda-builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt install -y \
    build-essential \
    cmake \
    curl \
    gcc  \
    g++ \
    git \
    git-lfs \
    libssl-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-setuptools \
    tar \
    wget

ENV TGI_INSTALL_PREFIX=/usr/local/tgi
ENV TENSORRT_INSTALL_PREFIX=/usr/local/tensorrt

# Install OpenMPI
FROM cuda-builder AS mpi-builder
ARG OMPI_VERSION

ENV OMPI_TARBALL_FILENAME="openmpi-$OMPI_VERSION.tar.bz2"
RUN wget "https://download.open-mpi.org/release/open-mpi/v4.1/$OMPI_TARBALL_FILENAME" -P /opt/src && \
    mkdir /usr/src/mpi && \
    tar -xf "/opt/src/$OMPI_TARBALL_FILENAME" -C /usr/src/mpi --strip-components=1 && \
    cd /usr/src/mpi && \
    ./configure --prefix=/usr/local/mpi --with-cuda=/usr/local/cuda && \
    make -j all && \
    make install && \
    rm -rf "/opt/src/$OMPI_TARBALL_FILENAME"

# Install TensorRT
FROM cuda-builder AS trt-builder
COPY backends/trtllm/scripts/install_tensorrt.sh /opt/install_tensorrt.sh
RUN chmod +x /opt/install_tensorrt.sh && \
    /opt/install_tensorrt.sh

# Build Backend
FROM cuda-builder AS tgi-builder
WORKDIR /usr/src/text-generation-inference

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y && \
    chmod -R a+w /root/.rustup && \
    chmod -R a+w /root/.cargo

ENV PATH="/root/.cargo/bin:$PATH"
RUN cargo install cargo-chef

# Cache dependencies
COPY --from=planner /usr/src/text-generation-inference/backends/trtllm/recipe.json .
RUN cargo chef cook --release --recipe-path recipe.json

# Build actual TGI
ARG CUDA_ARCH_LIST
ENV CMAKE_PREFIX_PATH="/usr/local/mpi:/usr/local/tensorrt:$CMAKE_PREFIX_PATH"
ENV LD_LIBRARY_PATH="/usr/local/mpi/lib:$LD_LIBRARY_PATH"
ENV PKG_CONFIG_PATH="/usr/local/mpi/lib/pkgconfig:$PKG_CONFIG_PATH"

COPY . .
COPY --from=trt-builder /usr/local/tensorrt /usr/local/tensorrt
COPY --from=mpi-builder /usr/local/mpi /usr/local/mpi
RUN mkdir $TGI_INSTALL_PREFIX && mkdir "$TGI_INSTALL_PREFIX/include" && mkdir "$TGI_INSTALL_PREFIX/lib" && \
    cd backends/trtllm && \
    CMAKE_INSTALL_PREFIX=$TGI_INSTALL_PREFIX cargo build --release

FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04 AS runtime
WORKDIR /usr/local/tgi/bin

ENV LD_LIBRARY_PATH="/usr/local/tgi/lib:/usr/local/tensorrt/lib:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"

COPY --from=mpi-builder /usr/local/mpi /usr/local/mpi
COPY --from=trt-builder /usr/local/tensorrt /usr/local/tensorrt
COPY --from=tgi-builder /usr/local/tgi /usr/local/tgi
COPY --from=tgi-builder /usr/src/text-generation-inference/target/release/text-generation-backends-trtllm /usr/local/tgi/bin/text-generation-launcher

FROM runtime

LABEL co.huggingface.vendor="Hugging Face Inc."
LABEL org.opencontainers.image.authors="hardware@hf.co"

ENTRYPOINT ["./text-generation-launcher"]
CMD ["--executor-worker", "/usr/local/tgi/bin/executorWorker"]
