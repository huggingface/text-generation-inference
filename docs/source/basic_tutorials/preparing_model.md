# Preparing the Model

Text Generation Inference improves the model in several aspects. 

## Quantization

TGI supports [bits-and-bytes](https://github.com/TimDettmers/bitsandbytes#bitsandbytes) and [GPT-Q](https://arxiv.org/abs/2210.17323) quantization. To speed up inference with quantization, simply set `quantize` flag to `bitsandbytes` or `gptq` depending on the quantization technique you wish to use. 


## RoPE Scaling

RoPE scaling can be used to increase the sequence length of the model during the inference time without necessarily fine-tuning it. To enable RoPE scaling, simply set `ROPE_SCALING` and `ROPE_FACTOR` variables. `ROPE_SCALING` can take the values `linear` or `dynamic`. If your model is not fine-tuned to a longer sequence length, use `dynamic`. `ROPE_FACTOR` is the ratio between the intended max sequence length and the model's original max sequence length.

## Safetensors

[Safetensors](https://github.com/huggingface/safetensors) is a fast and safe persistence format for deep learning models. TGI supports `safetensors` model loading under the hood. By default, given a repository with `safetensors` and `pytorch` weights, TGI will always load `safetensors`. If there's no `pytorch` weights, TGI will convert the weights to `safetensors` format.