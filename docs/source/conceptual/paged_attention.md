# Paged Attention

LLMs struggle with memory limitations during generation. In the decoding part of generation, all input tokens generated keys and values are stored in GPU memory, also referred to as _KV cache_. KV cache is exhaustive for memory, which causes inefficiencies in LLM serving.

PagedAttention addresses the memory waste by partitioning the KV cache into blocks, allowing keys and values to be stored in non-contiguous memory. This approach improves GPU utilization and throughput.

PagedAttention keeps a block table for memory sharing. This enables e.g. parallel sampling, where for a given prompt, multiple outputs are generated, and the computation and memory are shared between the outputs.

You can learn more about PagedAttention by reading the documentation [here](https://vllm.ai/).
