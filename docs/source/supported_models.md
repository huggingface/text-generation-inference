# Supported Models and Hardware

Text Generation Inference enables serving optimized models on specific hardware for the highest performance. The following sections list which models are hardware are supported.

## Supported Models

The following models are optimized and can be served with TGI, which uses custom CUDA kernels for better inference. You can add the flag `--disable-custom-kernels` at the end of the `docker run` command if you wish to disable them.

- [BLOOM](https://huggingface.co/bigscience/bloom)
- [FLAN-T5](https://huggingface.co/google/flan-t5-xxl)
- [Galactica](https://huggingface.co/facebook/galactica-120b)
- [GPT-2](https://huggingface.co/openai-community/gpt2)
- [GPT-Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [Llama](https://github.com/facebookresearch/llama)
- [OPT](https://huggingface.co/facebook/opt-66b)
- [SantaCoder](https://huggingface.co/bigcode/santacoder)
- [Starcoder](https://huggingface.co/bigcode/starcoder)
- [Falcon 7B](https://huggingface.co/tiiuae/falcon-7b)
- [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b)
- [MPT](https://huggingface.co/mosaicml/mpt-30b)
- [Llama V2](https://huggingface.co/meta-llama)
- [Code Llama](https://huggingface.co/codellama)
- [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [Phi](https://huggingface.co/microsoft/phi-2)
- [Idefics](HuggingFaceM4/idefics-9b-instruct) (Multimodal)
- [Llava-next](llava-hf/llava-v1.6-mistral-7b-hf) (Multimodal)

If the above list lacks the model you would like to serve, depending on the model's pipeline type, you can try to initialize and serve the model anyways to see how well it performs, but performance isn't guaranteed for non-optimized models:

```python
# for causal LMs/text-generation models
AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")`
# or, for text-to-text generation models
AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")
```

If you wish to serve a supported model that already exists on a local folder, just point to the local folder.

```bash
text-generation-launcher --model-id <PATH-TO-LOCAL-BLOOM>
``````


## Supported Hardware

TGI optimized models are supported on NVIDIA [A100](https://www.nvidia.com/en-us/data-center/a100/), [A10G](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) and [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPUs with CUDA 12.2+. Note that you have to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to use it. For other NVIDIA GPUs, continuous batching will still apply, but some operations like flash attention and paged attention will not be executed.

TGI also has support of ROCm-enabled AMD Instinct MI210 and MI250 GPUs, with paged attention, GPTQ quantization, flash attention v2 support. The following features are currently not supported in the ROCm version of TGI, and the supported may be extended in the future:
* Loading [AWQ](https://huggingface.co/docs/transformers/quantization#awq) checkpoints.
* Flash [layer norm kernel](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm)
* Kernel for sliding window attention (Mistral)

TGI is also supported on the following AI hardware accelerators:
- *Habana first-gen Gaudi and Gaudi2:* check out this [repository](https://github.com/huggingface/tgi-gaudi) to serve models with TGI on Gaudi and Gaudi2 with [Optimum Habana](https://huggingface.co/docs/optimum/habana/index)
* *AWS Inferentia2:* check out this [guide](https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference) on how to serve models with TGI on Inferentia2.
