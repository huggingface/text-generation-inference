# Flash Attention

Scaling transformer architecture is heavily bottlenecked by self-attention mechanism, which has quadratic time and memory complexity. Recent developments in accelerator hardware are mainly focused on enhancing compute capacities, and not memory and transferring data between hardware. This results in attention operation to have a bottleneck in memory, also called as _memory-bound_. Flash Attention is an attention algorithm used to overcome this problem and scale transformer based models more efficiently, enabling faster training and inference. 
In standard attention implementation, cost of loading and writing key, query, values from High Bandwidth Memory (HBM) is high. It loads key, query, value from HBM to GPU, performs each singular step of attention mechanism and writes it back to HBM repeatedly. Instead, Flash Attention loads Q, K, V once, fuses the operations in attention mechanism and writes it back. 
It is implemented for models with custom kernels, you can check out full list of models that support Flash Attention [here](https://github.com/huggingface/text-generation-inference/tree/main/server/text_generation_server/models). 


