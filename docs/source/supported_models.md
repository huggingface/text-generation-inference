
# Supported Models

Text Generation Inference enables serving optimized models. The following sections list which models (VLMs & LLMs) are supported.

- [Deepseek V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)
- [Deepseek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b) (Multimodal)
- [Idefics 3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3) (Multimodal)
- [Llava Next (1.6)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) (Multimodal)
- [Llama](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
- [Llama4](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
- [Phi 3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Granite](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct)
- [Gemma](https://huggingface.co/google/gemma-7b)
- [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224)
- [Gemma2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [Gemma3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)
- [Gemma3 Text](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)
- [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
- [Dbrx](https://huggingface.co/databricks/dbrx-instruct)
- [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)
- [Mistral](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
- [Gpt Bigcode](https://huggingface.co/bigcode/gpt_bigcode-santacoder)
- [Phi](https://huggingface.co/microsoft/phi-1_5)
- [PhiMoe](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
- [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)
- [StarCoder 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)
- [Qwen 2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)
- [Qwen 2 VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [Qwen 2.5 VL](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [Opt](https://huggingface.co/facebook/opt-6.7b)
- [T5](https://huggingface.co/google/flan-t5-xxl)
- [Galactica](https://huggingface.co/facebook/galactica-120b)
- [SantaCoder](https://huggingface.co/bigcode/santacoder)
- [Bloom](https://huggingface.co/bigscience/bloom-560m)
- [Mpt](https://huggingface.co/mosaicml/mpt-7b-instruct)
- [Gpt2](https://huggingface.co/openai-community/gpt2)
- [Gpt Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [Gptj](https://huggingface.co/EleutherAI/gpt-j-6b)
- [Idefics](https://huggingface.co/HuggingFaceM4/idefics-9b) (Multimodal)
- [Mllama](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) (Multimodal)



If the above list lacks the model you would like to serve, depending on the model's pipeline type, you can try to initialize and serve the model anyways to see how well it performs, but performance isn't guaranteed for non-optimized models:

```python
# for causal LMs/text-generation models
AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")
# or, for text-to-text generation models
AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")
```

If you wish to serve a supported model that already exists on a local folder, just point to the local folder.

```bash
text-generation-launcher --model-id <PATH-TO-LOCAL-BLOOM>
```
