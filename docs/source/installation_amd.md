# Using TGI with AMD GPUs

TGI is supported and tested on [AMD Instinct MI210](https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html), [MI250](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html) and [MI300](https://www.amd.com/en/products/accelerators/instinct/mi300.html) GPUs. The support may be extended in the future. The recommended usage is through Docker. Make sure to check the [AMD documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) on how to use Docker with AMD GPUs.

On a server powered by AMD GPUs, TGI can be launched with the following command:

```bash
model=teknium/OpenHermes-2.5-Mistral-7B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 256g --net host -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.0.4-rocm \
    --model-id $model
```

The launched TGI server can then be queried from clients, make sure to check out the [Consuming TGI](./basic_tutorials/consuming_tgi) guide.

## TunableOp

TGI's docker image for AMD GPUs integrates [PyTorch's TunableOp](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable), which allows to do an additional warmup to select the best performing matrix multiplication (GEMM) kernel from rocBLAS or hipBLASLt.

Experimentally, on MI300X, we noticed a 6-8% latency improvement when using TunableOp on top of ROCm 6.1 and PyTorch 2.3.

TunableOp is enabled by default, the warmup may take 1-2 minutes. In case you would like to disable TunableOp, please pass `--env PYTORCH_TUNABLEOP_ENABLED="0"` when launcher TGI's docker container.

## Flash attention implementation

Two implementations of Flash Attention are available for ROCm, the first is [ROCm/flash-attention](https://github.com/ROCm/flash-attention) based on a [Composable Kernel](https://github.com/ROCm/composable_kernel) (CK) implementation, and the second is a [Triton implementation](https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/utils/flash_attn_triton.py).

By default, the Composable Kernel implementation is used. However, the Triton implementation has slightly lower latency on MI250 and MI300, but requires a warmup which can be prohibitive as it needs to be done again for each new prompt length. If needed, FA Triton impelmentation can be enabled with `--env ROCM_USE_FLASH_ATTN_V2_TRITON="0"` when launching TGI's docker container.

## Unsupported features

The following features are currently not supported in the ROCm version of TGI, and the supported may be extended in the future:
* Loading [AWQ](https://huggingface.co/docs/transformers/quantization#awq) checkpoints.
* Kernel for sliding window attention (Mistral)
