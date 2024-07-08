# Installation from source

<Tip warning={true}>

Installing TGI from source is not the recommended usage. We strongly recommend to use TGI through Docker, check the [Quick Tour](./quicktour), [Installation for Nvidia GPUs](./installation_nvidia) and [Installation for AMD GPUs](./installation_amd) to learn how to use TGI with Docker.

</Tip>

## Install CLI

You can use TGI command-line interface (CLI) to download weights, serve and quantize models, or get information on serving parameters.

To install the CLI, you need to first clone the TGI repository and then run `make`.

```bash
git clone https://github.com/huggingface/text-generation-inference.git && cd text-generation-inference
make install
```

If you would like to serve models with custom kernels, run

```bash
BUILD_EXTENSIONS=True make install
```

## Local Installation from Source

Before you start, you will need to setup your environment, and install Text Generation Inference. Text Generation Inference is tested on **Python 3.9+**.

Text Generation Inference is available on pypi, conda and GitHub.

To install and launch locally, first [install Rust](https://rustup.rs/) and create a Python virtual environment with at least
Python 3.9, e.g. using conda:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

conda create -n text-generation-inference python=3.9
conda activate text-generation-inference
```

You may also need to install Protoc.

On Linux:

```bash
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

On MacOS, using Homebrew:

```bash
brew install protobuf
```

Then run to install Text Generation Inference:

```bash
git clone https://github.com/huggingface/text-generation-inference.git && cd text-generation-inference
BUILD_EXTENSIONS=True make install
```

<Tip warning={true}>

On some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```bash
sudo apt-get install libssl-dev gcc -y
```

</Tip>

Once installation is done, simply run:

```bash
make run-falcon-7b-instruct
```

This will serve Falcon 7B Instruct model from the port 8080, which we can query.
