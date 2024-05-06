# Preparing the Model

Text Generation Inference improves the model in several aspects.

## Quantization

TGI supports [bits-and-bytes](https://github.com/TimDettmers/bitsandbytes#bitsandbytes), [GPT-Q](https://arxiv.org/abs/2210.17323) and [AWQ](https://arxiv.org/abs/2306.00978) quantization. To speed up inference with quantization, simply set `quantize` flag to `bitsandbytes`, `gptq` or `awq` depending on the quantization technique you wish to use. When using GPT-Q quantization, you need to point to one of the models [here](https://huggingface.co/models?search=gptq) when using AWQ quantization, you need to point to one of the models [here](https://huggingface.co/models?search=awq). To get more information about quantization, please refer to [quantization guide](./../conceptual/quantization)


## RoPE Scaling

RoPE scaling can be used to increase the sequence length of the model during the inference time without necessarily fine-tuning it. To enable RoPE scaling, simply pass `--rope-scaling`, `--max-input-length` and `--rope-factors` flags when running through CLI. `--rope-scaling` can take the values `linear` or `dynamic`. If your model is not fine-tuned to a longer sequence length, use `dynamic`. `--rope-factor` is the ratio between the intended max sequence length and the model's original max sequence length. Make sure to pass `--max-input-length` to provide maximum input length for extension.

<Tip>

We recommend using `dynamic` RoPE scaling.

</Tip>

## Safetensors

[Safetensors](https://github.com/huggingface/safetensors) is a fast and safe persistence format for deep learning models, and is required for tensor parallelism. TGI supports `safetensors` model loading under the hood. By default, given a repository with `safetensors` and `pytorch` weights, TGI will always load `safetensors`. If there's no `pytorch` weights, TGI will convert the weights to `safetensors` format.
