# Installation

This section explains how to install the CLI tool as well as installing TGI from source. **The strongly recommended approach is to use Docker, as it does not require much setup. Check [the Quick Tour](./quicktour) to learn how to run TGI with Docker.**

## Install CLI

You can use TGI command-line interface (CLI) to download weights, serve and quantize models, or get information on serving parameters. 

To install TGI to use with CLI, you need to first clone the TGI repository, then inside the repository, run

```shell
git clone https://github.com/huggingface/text-generation-inference.git && cd text-generation-inference
make install
```

If you would like to serve models with custom kernels, run

```shell
BUILD_EXTENSIONS=True make install
```

## Running CLI

After installation, you will be able to use `text-generation-server` and `text-generation-launcher`.

`text-generation-server` lets you download the model with `download-weights` command like below 👇 

```shell
text-generation-server download-weights MODEL_HUB_ID
```

You can also use it to quantize models like below 👇 

```shell
text-generation-server quantize MODEL_HUB_ID OUTPUT_DIR 
```

You can use `text-generation-launcher` to serve models. 

```shell
text-generation-launcher --model-id MODEL_HUB_ID --port 8080
```

There are many options and parameters you can pass to `text-generation-launcher`. The documentation for CLI is kept minimal and intended to rely on self-generating documentation, which can be found by running 

```shell
text-generation-launcher --help
``` 

You can also find it hosted in this [Swagger UI](https://huggingface.github.io/text-generation-inference/).

Same documentation can be found for `text-generation-server`.

```shell
text-generation-server --help
```

## Local Installation from Source

Before you start, you will need to setup your environment, and install Text Generation Inference. Text Generation Inference is tested on **Python 3.9+**.

Text Generation Inference is available on pypi, conda and GitHub. 

To install and launch locally, first [install Rust](https://rustup.rs/) and create a Python virtual environment with at least
Python 3.9, e.g. using conda:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

conda create -n text-generation-inference python=3.9
conda activate text-generation-inference
```

You may also need to install Protoc.

On Linux:

```shell
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

On MacOS, using Homebrew:

```shell
brew install protobuf
```

Then run to install Text Generation Inference:

```shell
git clone https://github.com/huggingface/text-generation-inference.git && cd text-generation-inference
BUILD_EXTENSIONS=True make install
```

<Tip warning={true}>

On some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```shell
sudo apt-get install libssl-dev gcc -y
```

</Tip>

Once installation is done, simply run:

```shell
make run-falcon-7b-instruct
```

This will serve Falcon 7B Instruct model from the port 8080, which we can query.
