# Installation

This section explains how to install the CLI tool as well as installing TGI from source. **The strongly recommended approach is to use Docker, as it does not require much setup. Check [the Quick Tour](./quicktour) to learn how to run TGI with Docker.**

## Install CLI

TODO


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
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels
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

To see all options to serve your models, check in the [codebase](https://github.com/huggingface/text-generation-inference/blob/main/launcher/src/main.rs) or the CLI:

```bash
text-generation-launcher --help
```
