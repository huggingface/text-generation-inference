# PagedAttention

LLMs struggle with memory limitations during generation. In the decoding part of generation, all the attention keys and values generated for previous tokens are stored in GPU memory for reuse. This is called _KV cache_, and it may take up a large amount of memory for large models and long sequences.

PagedAttention attempts to optimize memory use by partitioning the KV cache into blocks that are accessed through a lookup table. Thus, the KV cache does not need to be stored in contiguous memory, and blocks are allocated as needed. The memory efficiency can increase GPU utilization on memory-bound workloads, so more inference batches can be supported.

The use of a lookup table to access the memory blocks can also help with KV sharing across multiple generations. This is helpful for techniques such as _parallel sampling_, where multiple outputs are generated simultaneously for the same prompt. In this case, the cached KV blocks can be shared among the generations.

TGI's PagedAttention implementation leverages the custom cuda kernels developed by the [vLLM Project](https://github.com/vllm-project/vllm). You can learn more about this technique in the [project's page](https://vllm.ai/).
