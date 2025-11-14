# Text-generation-inference - Neuron backend for AWS Trainium and inferentia2

## Description

This is the TGI backend for AWS Neuron Trainium and Inferentia family of chips.

This backend is composed of:
- the AWS Neuron SDK,
- the legacy v2 TGI launcher and router,
- a neuron specific inference server for text-generation.

## Usage

Please refer to the official [documentation](https://huggingface.co/docs/text-generation-inference/backends/neuron).

## Build your own image

The simplest way to build TGI with the neuron backend is to use the provided `Makefile`:

```shell
$ make -C backends/neuron image
```

Alternatively, you can build the image directly from the top directory using a command similar to the one defined
in the `Makefile` under the `image` target.
