# Supported Models and Hardware

## Supported Models

List of optimized models are below.

- [BLOOM](https://huggingface.co/bigscience/bloom)
- [FLAN-T5](https://huggingface.co/google/flan-t5-xxl)
- [Galactica](https://huggingface.co/facebook/galactica-120b)
- [GPT-Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [Llama](https://github.com/facebookresearch/llama)
- [OPT](https://huggingface.co/facebook/opt-66b)
- [SantaCoder](https://huggingface.co/bigcode/santacoder)
- [Starcoder](https://huggingface.co/bigcode/starcoder)
- [Falcon 7B](https://huggingface.co/tiiuae/falcon-7b)
- [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b)
- [MPT](https://huggingface.co/mosaicml/mpt-30b)
- [Llama V2](https://huggingface.co/meta-llama)

If the above list lacks the model you would like to serve, depending on the model's pipeline type, you can try to initialize and serve the model on best-effort basis like below:

`AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")`

or

`AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")`. 

For the optimized models above, TGI uses custom CUDA kernels for better inference. You can add the flag `--disable-custom-kernels` at the end of the `docker run` command if you wish to disable them.


## Supported Hardware

Text Generation Inference optimized models supported on NVIDIA [A100](https://www.nvidia.com/en-us/data-center/a100/), [A10G](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) and [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPUs with CUDA 11.8+. Note that you have to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to use it. For other hardware, continuous batching will still apply, but you might observe downgrades on some of the operations (e.g. flash attention, paged attention) will not be executed. 

TGI is also supported on the following AI hardware accelerators:
- *Habana first-gen Gaudi and Gaudi2:* checkout [here](https://github.com/huggingface/optimum-habana/tree/main/text-generation-inference) how to serve models with TGI on Gaudi and Gaudi2 with [Optimum Habana](https://huggingface.co/docs/optimum/habana/index)



