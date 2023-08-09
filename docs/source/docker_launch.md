# Launching with Docker

The easiest way of getting started is using the official Docker container. Install Docker following [their installation instructions](https://docs.docker.com/get-docker/).

Let's say you want to deploy [Falcon-7B Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) model with TGI. Here is an example on how to do that:

```shell
model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.0 --model-id $model
```

<Tip warning={true}>

To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)  . We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

</Tip>

To see all possible flags and options, you can use the `--help` flag. It's possible to configure the number of shards, quantization, generation parameters, and more.

```
docker run ghcr.io/huggingface/text-generation-inference:1.0.0 --help
```