# Paged Attention

LLMs struggle with memory limitations during generation. In the decoding part of generation, all input tokens generated keys and values are stored in GPU memory, also referred as _KV cache_. KV cache is exhaustive for memory which causes inefficiencies in LLM serving.

PagedAttention addresses the memory waste by partitioning the KV cache into blocks, allowing keys and values to be stored in non-contiguous memory. This approach improves GPU utilization and throughput.

PagedAttention also enables memory sharing, useful for parallel sampling. PagedAttention keeps track of shared memory through a block table and implements the Copy-on-Write mechanism to ensure safe sharing.

You can learn more about PagedAttention by reading the documentation [here](https://vllm.ai/).