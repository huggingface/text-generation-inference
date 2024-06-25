# FP8 (fp8_e4m3) KV Cache Scaling Factor Utility

This utility is provided to generate model with `FP8(fp8_e4m3)` quantized KV cache scales. The generated scaling factors are then saved to the corresponding HF model, which can be used with Text Generation Inference (TGI).

The KV scales are integrated into the HF model in the following format. The FP8 KV cache scaling factors are specified through the `.kv_scale` parameter within the `Attention` module, as shown below:


```
model.layers.0.self_attn.kv_scale                < F32
model.layers.1.self_attn.kv_scale                < F32
...
```

Additionally, `kv_cache_torch_dtype` attribute is added to `config.json` which indicates the torch dtype (`float8_e4m3fn` in this utility) used to generate scales.

Example config: [Llama-2-7b-chat-hf-FP8-KV#config.json](https://huggingface.co/mohitsha/Llama-2-7b-chat-hf-FP8-KV/blob/main/config.json#L14)

Note: The utility supports only a selected LLAMA type models. Please adapt the script for other models.

## Prerequisites

- Nvidia AMMO (nvidia-ammo==0.7.1)
- Hugging Face Transformers

```bash
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo==0.7.1
```

## CLI options
```
usage: create_fp8_kv_scales_model.py [-h] --model_dir MODEL_DIR [--device DEVICE] [--dtype DTYPE] [--batch_size BATCH_SIZE] [--calib_size CALIB_SIZE] [--output_dir OUTPUT_DIR]

Adapted from examples/quantization/hf_ptq.py

options:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Specify where the HuggingFace model is
  --device DEVICE
  --dtype DTYPE         Model data type.
  --batch_size BATCH_SIZE
                        Batch size for calibration.
  --calib_size CALIB_SIZE
                        Number of samples for calibration.
  --output_dir OUTPUT_DIR

```

## Example usage
```
python create_fp8_kv_scales_model.py --model_dir meta-llama/Llama-2-70b-chat-hf --output_dir output
```
