# Safetensors

Safetensors is a model serialization format for deep learning models. It is [faster](https://huggingface.co/docs/safetensors/speed) and safer compared to other serialization formats like pickle (which is used under the hood in many deep learning libraries).

TGI depends on safetensors format mainly to enable [tensor parallelism sharding](./tensor_parallelism). For a given model repository during serving, TGI looks for safetensors weights. If there are no safetensors weights, TGI converts the PyTorch weights to safetensors format.

You can learn more about safetensors by reading the [safetensors documentation](https://huggingface.co/docs/safetensors/index).
