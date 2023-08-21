# Flash Attention

Scaling transformer architecture is heavily bottlenecked by the self-attention mechanism, which has quadratic time and memory complexity. Recent developments in accelerator hardware are mainly focused on enhancing compute capacities and not memory and transferring data between hardware. This results in attention operation having a bottleneck in memory, also called _memory-bound_. Flash Attention is an attention algorithm used to reduce this problem and scale transformer-based models more efficiently, enabling faster training and inference. 
In standard attention implementation, the cost of loading and writing keys, queries, and values from High Bandwidth Memory (HBM) is high. It loads key, query, value from HBM to GPU, performs a single step of the attention mechanism and writes it back to HBM, and repeats this for every singular step of the attention. Instead, Flash Attention loads keys, queries, and values once, fuses the operations of the attention mechanism and writes them back. 
It is implemented for models with custom kernels, you can check out the full list of models that support Flash Attention [here](https://github.com/huggingface/text-generation-inference/tree/main/server/text_generation_server/models). 


