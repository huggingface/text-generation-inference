
# Supported Models and Hardware

Text Generation Inference enables serving optimized models on specific hardware for the highest performance. The following sections list which models are hardware are supported.

## Supported Models

- [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b) (Multimodal)
- [Llava Next (1.6)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) (Multimodal)
- [Llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Phi 3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Gemma](https://huggingface.co/google/gemma-7b)
- [Gemma2](https://huggingface.co/google/gemma2-9b)
- [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
- [Dbrx](https://huggingface.co/databricks/dbrx-instruct)
- [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)
- [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
- [Gpt Bigcode](https://huggingface.co/bigcode/gpt_bigcode-santacoder)
- [Phi](https://huggingface.co/microsoft/phi-1_5)
- [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)
- [StarCoder 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)
- [Qwen 2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)
- [Opt](https://huggingface.co/facebook/opt-6.7b)
- [T5](https://huggingface.co/google/flan-t5-xxl)
- [Galactica](https://huggingface.co/facebook/galactica-120b)
- [SantaCoder](https://huggingface.co/bigcode/santacoder)
- [Bloom](https://huggingface.co/bigscience/bloom-560m)
- [Mpt](https://huggingface.co/mosaicml/mpt-7b-instruct)
- [Gpt2](https://huggingface.co/openai-community/gpt2)
- [Gpt Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [Idefics](https://huggingface.co/HuggingFaceM4/idefics-9b) (Multimodal)


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
```
