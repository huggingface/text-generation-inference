# Text Generation Inference

Text-Generation-Inference is, an open-source, purpose-built solution for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation using Tensor Parallelism and dynamic batching for the most popular open-source LLMs, including StarCoder, BLOOM, GPT-NeoX, Llama, and T5. Text Generation Inference implements optimization for all supported model architectures, including:

- Tensor Parallelism and custom cuda kernels
- Optimized transformers code for inference using flash-attention and Paged Attention on the most popular architectures
- Quantization with bitsandbytes or gptq
- Continuous batching of incoming requests for increased total throughput
- Accelerated weight loading (start-up time) with safetensors
- Logits warpers (temperature scaling, topk, repetition penalty ...)
- Watermarking with A Watermark for Large Language Models
- Stop sequences, Log probabilities
- Token streaming using Server-Sent Events (SSE)

