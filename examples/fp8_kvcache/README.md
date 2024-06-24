# FP8 (fp8_e4m3) KV Cache Scaling Factor Extraction Utility

This utility is designed to extract KV cache scaling factors from a quantized `FP8(fp8_e4m3)` Hugging Face (HF) model. The extracted scaling factors are then saved to the corresponding unquantized HF model, which can be used with Text Generation Inference (TGI).

Note: This tool specifically works with models quantized using the [AutoFP8](https://github.com/neuralmagic/AutoFP8/tree/main) repository.

The KV scales are integrated into the unquantized HF model in the following format. The FP8 KV cache scaling factors are added to the FP16 checkpoints and specified through the .kv_scale parameter within the Attention module, as shown below:

```
model.layers.0.self_attn.kv_scale                < F32
model.layers.1.self_attn.kv_scale                < F32
...
```

## Prerequisites

- text-generation-server
- AutoFP8

## CLI options
```
usage: extract_fp8_kv_scales.py [-h] [--quantized-model QUANTIZED_MODEL] [--model MODEL] [--save-path SAVE_PATH]

Extract FP8 KV cache scales and add them to a FP16 model.

options:
  -h, --help            show this help message and exit
  --quantized-model QUANTIZED_MODEL
                        Path to the FP8 model checkpoint to extract KV cache scales
  --model MODEL         Model ID of the FP16 model to save the KV cache scales
  --save-path SAVE_PATH
                        Path to save the FP16 model with the kv scales
```

## Example usage
To extract KV cache scaling factors from a quantized FP8 model and save them to an unquantized FP16 model, use the following command:

```
python extract_fp8_kv_scales.py --quantized-model neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV --model meta-llama/Meta-Llama-3-8B-Instruct --save-path  Meta-Llama-3-8B-Instruct
```
