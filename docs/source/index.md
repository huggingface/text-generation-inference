# Text Generation Inference

Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and T5.

![Text Generation Inference](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/TGI.png)

Text Generation Inference implements many optimizations and features, such as:

- Simple launcher to serve most popular LLMs
- Production ready (distributed tracing with Open Telemetry, Prometheus metrics)
- Tensor Parallelism for faster inference on multiple GPUs
- Token streaming using Server-Sent Events (SSE)
- Continuous batching of incoming requests for increased total throughput
- Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention) and [Paged Attention](https://github.com/vllm-project/vllm) on the most popular architectures
- Quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPT-Q](https://arxiv.org/abs/2210.17323)
- [Safetensors](https://github.com/huggingface/safetensors) weight loading
- Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
- Logits warper (temperature scaling, top-p, top-k, repetition penalty)
- Stop sequences
- Log probabilities
- Fine-tuning Support: Utilize fine-tuned models for specific tasks to achieve higher accuracy and performance.
- [Guidance](../conceptual/guidance): Enable function calling and tool-use by forcing the model to generate structured outputs based on your own predefined output schemas.

Text Generation Inference is used in production by multiple projects, such as:

- [Hugging Chat](https://github.com/huggingface/chat-ui), an open-source interface for open-access models, such as Open Assistant and Llama
- [OpenAssistant](https://open-assistant.io/), an open-source community effort to train LLMs in the open
- [nat.dev](http://nat.dev/), a playground to explore and compare LLMs.
