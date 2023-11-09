# Text Generation Inference chart

Helm chart for deploying [Text Generation Inference](https://huggingface.co/docs/text-generation-inference) to Kubernetes.

## Installation
### Starcoder

Here is an example of the values to pass to the chart in order to deploy [bigcode/starcoderbase-7b](https://huggingface.co/bigcode/starcoderbase-7b)
```yaml
---
args:
  - "--model-id"
  - "bigcode/starcoderbase-7b"
  - "--num-shard"
  - "1"

env:
  HUGGING_FACE_HUB_TOKEN: hf_FIXME

persistence:
  storageClassName: "default"
  accessModes: ["ReadWriteOnce"]
  storage: 150Gi
```
```shell
helm install -f values.yaml startcoder .
```
