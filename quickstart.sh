#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  # Install Docker based on the OS
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Debian-based systems
    sudo apt-get update && sudo apt-get install -y docker.io
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (using Homebrew)
    brew install --cask docker
  elif [[ "$OSTYPE" == "msys"* ]]; then
    # Windows (using Chocolatey)
    choco install docker-desktop
  else
    echo "Unsupported OS: $OSTYPE"
    exit 1
  fi
fi

# Install NVIDIA Container Toolkit (only if on Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # Currently, NVIDIA Container Toolkit does not support macOS, so warn the user.
  echo "NVIDIA Container Toolkit is not available for macOS."
  exit 1
elif [[ "$OSTYPE" == "msys"* ]]; then
  # Currently, NVIDIA Container Toolkit does not support Windows directly.
  echo "NVIDIA Container Toolkit is not available for Windows. Ensure Docker Desktop WSL 2 backend is enabled."
fi

# Set variables for the Docker container
model="HuggingFaceH4/zephyr-7b-beta"
volume="$PWD/data"

# Run the Docker container in interactive mode to allow CTRL+C to stop the container
docker run -it --gpus all --shm-size 1g -p 8080:80 -v "$volume:/data" ghcr.io/huggingface/text-generation-inference:2.0 --model-id "$model"
