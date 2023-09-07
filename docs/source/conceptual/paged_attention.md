# PagedAttention

LLMs struggle with memory limitations during generation. In the decoding part of generation, all the attention keys and values generated for previous tokens are stored in GPU memory for reuse. This is called _KV cache_, and it may take up a large amount of memory for large models and long sequences.

PagedAttention addresses the memory waste by partitioning the KV cache into blocks, allowing keys and values to be stored in non-contiguous memory. This approach improves GPU utilization and throughput.

PagedAttention keeps a block table for memory sharing. This enables e.g. parallel sampling, where for a given prompt, multiple outputs are generated, and the computation and memory are shared between the outputs.

You can learn more about PagedAttention by reading the documentation [here](https://vllm.ai/).
