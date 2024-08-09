# Quantization

TGI offers many quantization schemes to run LLMs effectively and fast based on your use-case. TGI supports GPTQ, AWQ, bits-and-bytes, EETQ, Marlin, EXL2 and fp8 quantization.

To leverage GPTQ, AWQ, Marlin and EXL2 quants, you must provide pre-quantized weights. Whereas for bits-and-bytes, EETQ and fp8, weights are quantized by TGI on the fly.

We recommend using the official quantization scripts for creating your quants:
1. [AWQ](https://github.com/casper-hansen/AutoAWQ/blob/main/examples/quantize.py)
2. [GPTQ/ Marlin](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/examples/quantization/basic_usage.py)
3. [EXL2](https://github.com/turboderp/exllamav2/blob/master/doc/convert.md)

For on-the-fly quantization you simply need to pass one of the supported quantization types and TGI takes care of the rest.

## Quantization with bitsandbytes, EETQ & fp8

bitsandbytes is a library used to apply 8-bit and 4-bit quantization to models. Unlike GPTQ quantization, bitsandbytes doesn't require a calibration dataset or any post-processing – weights are automatically quantized on load. However, inference with bitsandbytes is slower than GPTQ or FP16 precision.

8-bit quantization enables multi-billion parameter scale models to fit in smaller hardware without degrading performance too much.
In TGI, you can use 8-bit quantization by adding `--quantize bitsandbytes` like below 👇

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --quantize bitsandbytes
```

4-bit quantization is also possible with bitsandbytes. You can choose one of the following 4-bit data types: 4-bit float (`fp4`), or 4-bit `NormalFloat` (`nf4`). These data types were introduced in the context of parameter-efficient fine-tuning, but you can apply them for inference by automatically converting the model weights on load.

In TGI, you can use 4-bit quantization by adding `--quantize bitsandbytes-nf4` or `--quantize bitsandbytes-fp4` like below 👇

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --quantize bitsandbytes-nf4
```

You can get more information about 8-bit quantization by reading this [blog post](https://huggingface.co/blog/hf-bitsandbytes-integration), and 4-bit quantization by reading [this blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

Similarly you can use pass you can pass `--quantize eetq` or `--quantize fp8` for respective quantization schemes.

In addition to this, TGI allows creating GPTQ quants directly by passing the model weights and a calibration dataset.

## Quantization with GPTQ

GPTQ is a post-training quantization method to make the model smaller. It quantizes the layers by finding a compressed version of that weight, that will yield a minimum mean squared error like below 👇

Given a layer \\(l\\) with weight matrix \\(W_{l}\\) and layer input \\(X_{l}\\), find quantized weight \\(\\hat{W}_{l}\\):

$$({\hat{W}_{l}}^{*} = argmin_{\hat{W_{l}}} ||W_{l}X-\hat{W}_{l}X||^{2}_{2})$$


TGI allows you to both run an already GPTQ quantized model (see available models [here](https://huggingface.co/models?search=gptq)) or quantize a model of your choice using quantization script. You can run a quantized model by simply passing --quantize like below 👇

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --quantize gptq
```

Note that TGI's GPTQ implementation doesn't use [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) under the hood. However, models quantized using AutoGPTQ or Optimum can still be served by TGI.

To quantize a given model using GPTQ with a calibration dataset, simply run

```bash
text-generation-server quantize tiiuae/falcon-40b /data/falcon-40b-gptq
# Add --upload-to-model-id MYUSERNAME/falcon-40b to push the created model to the hub directly
```

This will create a new directory with the quantized files which you can use with,

```bash
text-generation-launcher --model-id /data/falcon-40b-gptq/ --sharded true --num-shard 2 --quantize gptq
```

You can learn more about the quantization options by running `text-generation-server quantize --help`.

If you wish to do more with GPTQ models (e.g. train an adapter on top), you can read about transformers GPTQ integration [here](https://huggingface.co/blog/gptq-integration).
You can learn more about GPTQ from the [paper](https://arxiv.org/pdf/2210.17323.pdf).
