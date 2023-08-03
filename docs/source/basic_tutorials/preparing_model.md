# Preparing the Model

Text Generation Inference improves the model in several aspects. 

## Quantization

TGI supports `bits-and-bytes` and `GPT-Q` quantization. To speed up inference with quantization, simply set `quantize` flag to `bitsandbytes` or `gptq` depending on the quantization technique you wish to use. 


## RoPE Scaling

RoPE scaling can be used to increase the sequence length of the model during the inference time without necessarily fine-tuning it. To enable RoPE scaling, simply set `ROPE_SCALING` and `ROPE_FACTOR` variables. `ROPE_SCALING` can take the values `linear` or `dynamic`. If your model is not fine-tuned to a longer sequence length, use `dynamic`. `ROPE_FACTOR` is the ratio between the intended max sequence length and the model's original max sequence length.

## Safetensors Conversion

