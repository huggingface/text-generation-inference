# PagedAttention

LLMs struggle with memory limitations during generation. In the decoding part of generation, all the attention keys and values generated for previous tokens are stored in GPU memory for reuse. This is called _KV cache_, and it may take up a large amount of memory for large models and long sequences.

PagedAttention addresses the memory waste by partitioning the KV cache into blocks, allowing keys and values to be stored in non-contiguous memory. This approach improves GPU utilization and throughput.

The use of a lookup table to access the memory blocks can also help with KV sharing across multiple generations. This is helpful for techniques such as _parallel sampling_, where multiple outputs are generated simultaneously for the same prompt. In this case, the cached KV blocks can be shared among the generations.

TGI's PagedAttention implementation leverages the custom cuda kernels developed by the [vLLM Project](https://github.com/vllm-project/vllm). You can learn more about this technique in the [project's page](https://vllm.ai/).
