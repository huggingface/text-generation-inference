# Accelerating Inference with FP8 KV Cache

Text Generation Inference (TGI) now supports FP8 KV Cache, enhancing inference speed on both Nvidia and AMD GPUs. This feature significantly boosts performance and memory efficiency, enabling faster and more scalable text generation.

By quantizing the KV cache to 8-bit floating point (FP8) formats, we can greatly reduce the memory footprint. This reduction allows for:
* Increased token storage capacity in the cache
* Improved throughput in text generation tasks
* More efficient GPU memory utilization

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

* Nvidia GPUs:  Supports both FP8E4M3 and FP8E5M2.
* AMD GPUs: Supports FP8E4M3.

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
The FP8 kv cache scaling factors required in the FP16 checkpoints are specified through the .kv_scale parameter present on the `Attention` module, such as:

```
model.layers.0.self_attn.kv_scale                < F32
model.layers.1.self_attn.kv_scale                < F32
...
```

### Generating model with KV Cache scales

Use [AutoFP8](https://github.com/neuralmagic/AutoFP8) with calibration data to generate per-tensor scales for FP8 quantized KV Cache. For more details, see the following example: https://github.com/neuralmagic/AutoFP8/blob/main/example_dataset.py

TGI provides a utility to extract the FP8 KV cache scales from an `AutoFP8` quantized model and save them to the FP16 model for use with TGI. For more information: <path to script>

Alternatively, you can use other quantizer tools, such as Nvidia AMMO, to obtain these scaling factors.