# Accelerating Inference with FP8 KV Cache

Text Generation Inference (TGI) supports FP8 KV Cache, enhancing inference speed on both Nvidia and AMD GPUs.

FP8 KV Cache enhances the efficiency of text generation by quantizing the KV cache to FP8 format. Quantizing the KV cache to FP8 reduces its memory footprint, enabling storage of more tokens in cache. This improves overall throughput in text generation tasks.

In FP8 KV Cache, while the KV cache is stored in quantized FP8 format for memory efficiency, computations are performed in FP16 format. This strategy strikes a balance between conserving memory and maintaining computational accuracy.

## FP8 Formats: E4M3 and E5M2
The Open Compute Project (OCP) defines two common 8-bit floating point data formats:

E4M3:

* 1 sign bit
* 4 biased exponent bits
* 3 mantissa bits

E5M2:

* 1 sign bit
* 5 biased exponent bits
* 2 mantissa bits

E4M3 offers higher precision for representing floating point numbers. However, due to its limited range, E4M3 typically requires a higher-precision (usually FP32) scaling factor alongside each quantized tensor. Currently, TGI supports only per-tensor (scalar) scaling factors.

## Current Hardware Support

* Nvidia GPUs:  Supports both FP8E4M3 (fp8) and FP8E5M2 (fp8_e5m2).
* AMD GPUs: Supports FP8E4M3FNUZ (fp8).

## FP8 E5M2 KV Cache
Example usage:
```
text-generation-launcher --model-id <> --kv-cache-dtype fp8_e5m2
```

## FP8 E4M3 KV Cache
While E4M3 offers higher precision, it requires careful handling of scaling factors to maintain accuracy. Therefore, it is recommended to provide KV cache scaling factors as part of the FP16 checkpoint. If scaling factors are not provided, a default factor of 1.0 is used, which may lead to accuracy loss.

Example usage:
```
text-generation-launcher --model-id <> --kv-cache-dtype fp8
```

### Checkpoint structure for KV scales
The FP8 kv cache scaling factors, required in the model, are specified through the `.kv_scale` parameter present in the `Attention` module, such as:

```
model.layers.0.self_attn.kv_scale                < F32
model.layers.1.self_attn.kv_scale                < F32
...
```

When providing `.kv_scale` in model, the config should specify proper `kv_cache_torch_dtype` used to generate scales (`float8_e4m3fn` or `float8_e4m3fnuz`).

Example config: [Llama-2-7b-chat-hf-FP8-KV#config.json](https://huggingface.co/mohitsha/Llama-2-7b-chat-hf-FP8-KV/blob/main/config.json#L14)

### Generating model with KV Cache scales

TGI provides a utility to generate model with FP8 KV cache scales using Nvidia AMMO for use with TGI. For more information: [create_fp8_kv_scales_model.py](https://github.com/huggingface/text-generation-inference/examples/fp8_kvcache/create_fp8_kv_scales_model.py)

Alternatively, you can use other quantizer tools to obtain these scaling factors.
