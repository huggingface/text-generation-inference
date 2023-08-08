# Launching with Docker

The easiest way of getting started is using the official Docker container:

```shell
model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.0 --model-id $model
```

<Tip warning={true}>

To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)  . We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

</Tip>

To see all options to serve your models, check in the [codebase](https://github.com/huggingface/text-generation-inference/blob/main/launcher/src/main.rs) or the CLI:

```shell
text-generation-launcher --help
```
