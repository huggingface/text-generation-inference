# update system
apt-get update
apt-get upgrade -y

# install Linux tools and Python 3
apt-get install software-properties-common wget curl clang cmake git libopenmpi-dev libssl-dev \
    python3-dev python3-pip python3-wheel python3-setuptools -y

# install Python packages
python3 -m pip install --upgrade pip

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

# Install TensorRT
curl --proto '=https' --tlsv1.2 -sSf https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/common/install_tensorrt.sh | sh