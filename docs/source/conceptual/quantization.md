# Quantization

TGI offers GPTQ and bits-and-bytes quantization to quantize large language models.

## Quantization with GPTQ

GPTQ is a post-training quantization method to make the model smaller. It quantizes each weight by finding a compressed version of that weight, that will yield a minimum mean squared error like below 👇 

Given a layer \\(l\\) with weight matrix \\(W_{l}\\) and layer input \\(X_{l}\\), find quantized weight \\(\\hat{W}_{l}\\):

$$({\hat{W}_{l}}^{*} = argmin_{\hat{W_{l}}} ||W_{l}X-\hat{W}_{l}X||^{2}_{2})$$


TGI allows you to both run an already GPTQ quantized model (see available models [here](https://huggingface.co/models?search=gptq)) or quantize a model of your choice using quantization script by simply passing --quantize like below 👇 

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --quantize gptq
```

Note that TGI's GPTQ implementation is different than [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ). 

To run quantization only with a calibration dataset, simply run

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

## Quantization with bitsandbytes

bitsandbytes is a library used to apply 8-bit and 4-bit quantization to models. It can be used during training for mixed-precision training or before inference to make the model smaller. Unlike GPTQ quantization, bitsandbytes quantization doesn't require a calibration dataset or pre-quantized weights. One caveat of bitsandbytes 8-bit quantization is that the inference speed is slower compared to GPTQ or FP16 precision.

8-bit quantization enables multi-billion parameter scale models to fit in smaller hardware without degrading performance too much. 
In TGI, you can use 8-bit quantization by adding `--quantize bitsandbytes` like below 👇

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --quantize --bitsandbytes-nf4
```

4-bit Float (FP4) and 4-bit NormalFloat (NF4) are two data types introduced to use with QLoRA technique, a parameter-efficient fine-tuning technique. These data types can also be used to make a pre-trained model smaller. TGI essentially uses these data types to quantize an already trained model before the inference.

In TGI, you can use 4-bit quantization by adding `--quantize bitsandbytes-nf4` or `--quantize bitsandbytes-fp4` like below 👇 

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --quantize --bitsandbytes-nf4
```

You can get more information about 8-bit quantization by reading this [blog post](https://huggingface.co/blog/hf-bitsandbytes-integration), and 4-bit quantization by reading [this blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes).
