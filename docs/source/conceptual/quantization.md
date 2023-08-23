# Quantization

TGI offers GPTQ and bits-and-bytes quantization to quantize large language models.

## Quantization with GPTQ

GPTQ is a post-training quantization method to make the model smaller. It quantizes each weight by finding a compressed version of that weight, that will yield a minimum mean squared error like below 👇 

Given a layer \(l\) with weight matrix \(W_{l}\) and layer input \(X_{l}\), find quantized weight \(\hat{W}_{l}\):

\({\hat{W}{l}}^{*} = argmin{\hat{W_{l}}} |W_{l}X-\hat{W}{l}X|^{2}{2}\)

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