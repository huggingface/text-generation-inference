FROM ghcr.nju.edu.cn/huggingface/text-generation-inference:latest
MAINTAINER ailab

COPY . /code/
RUN  cd /code/ && \
     bash install_apt_source.sh && \
     export DEBIAN_FRONTEND=noninteractive && \
     apt update && \
     apt install -y wget git vim ssh libxml2 && \
     bash install_cuda.sh && \
     bash install_mpi.sh && \
     bash install_rust.sh && \
     bash install_tgi.sh && \
     bash bash_env.sh
