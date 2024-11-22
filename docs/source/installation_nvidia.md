# Using TGI with Nvidia GPUs

TGI optimized models are supported on NVIDIA [H100](https://www.nvidia.com/en-us/data-center/h100/), [A100](https://www.nvidia.com/en-us/data-center/a100/), [A10G](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) and [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPUs with CUDA 12.2+. Note that you have to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to use it.

For other NVIDIA GPUs, continuous batching will still apply, but some operations like flash attention and paged attention will not be executed.

TGI can be used on NVIDIA GPUs through its official docker image:

```bash
model=teknium/OpenHermes-2.5-Mistral-7B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 64g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.4.1 \
    --model-id $model
```

The launched TGI server can then be queried from clients, make sure to check out the [Consuming TGI](./basic_tutorials/consuming_tgi) guide.
