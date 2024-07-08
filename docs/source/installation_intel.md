# Using TGI with Intel GPUs

TGI optimized models are supported on Intel Data Center GPU [Max1100](https://www.intel.com/content/www/us/en/products/sku/232876/intel-data-center-gpu-max-1100/specifications.html), [Max1550](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html), the recommended usage is through Docker.


On a server powered by Intel GPUs, TGI can be launched with the following command:

```bash
model=teknium/OpenHermes-2.5-Mistral-7B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --rm --privileged --cap-add=sys_nice \
    --device=/dev/dri \
    --ipc=host --shm-size 1g --net host -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:latest-intel \
    --model-id $model --cuda-graphs 0
```

The launched TGI server can then be queried from clients, make sure to check out the [Consuming TGI](./basic_tutorials/consuming_tgi) guide.
