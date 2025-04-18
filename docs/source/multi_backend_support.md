# Multi-backend support

TGI (Text Generation Inference) offers flexibility by supporting multiple backends for serving large language models (LLMs).
With multi-backend support, you can choose the backend that best suits your needs,
whether you prioritize performance, ease of use, or compatibility with specific hardware. API interaction with
TGI remains consistent across backends, allowing you to switch between them seamlessly.

**Supported backends:**
* **TGI CUDA backend**: This high-performance backend is optimized for NVIDIA GPUs and serves as the default option
  within TGI. Developed in-house, it boasts numerous optimizations and is used in production by various projects, including those by Hugging Face.
* **[TGI TRTLLM backend](./backends/trtllm)**: This backend leverages NVIDIA's TensorRT library to accelerate LLM inference.
  It utilizes specialized optimizations and custom kernels for enhanced performance.
  However, it requires a model-specific compilation step for each GPU architecture.
* **[TGI Llamacpp backend](./backends/llamacpp)**: This backend facilitates the deployment of large language models
  (LLMs) by integrating [llama.cpp][llama.cpp], an advanced inference engine optimized for both CPU and GPU computation.
* **[TGI Neuron backend](./backends/neuron)**: This backend leverages the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) to allow the deployment of large language models (LLMs) on [AWS Trainium and Inferentia chips](https://aws.amazon.com/ai/machine-learning/trainium/).
