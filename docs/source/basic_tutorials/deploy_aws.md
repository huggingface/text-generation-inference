# Deploying TGI on AWS (EC2 and SageMaker)

This guide shows how to deploy **Text Generation Inference (TGI)** on AWS and how to benchmark it in a way that is useful for capacity planning.

## Deploy on EC2 (Docker)

For most setups, the simplest path is to run the official container on an EC2 GPU instance.

1. **Launch an EC2 GPU instance** (for example `g5.*` for NVIDIA GPUs).
2. **Install Docker + NVIDIA Container Toolkit** (see [Using TGI with Nvidia GPUs](../installation_nvidia) and NVIDIA’s installation docs).
3. **Run TGI**:

```bash
model=HuggingFaceH4/zephyr-7b-beta
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
  ghcr.io/huggingface/text-generation-inference:3.3.5 \
  --model-id "$model"
```

4. **Smoke test**:

```bash
curl 127.0.0.1:8080/generate \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"Hello","parameters":{"max_new_tokens":16}}'
```

## Deploy on SageMaker (real-time endpoint)

TGI includes a SageMaker compatibility route (`POST /invocations`) and a SageMaker entrypoint (`sagemaker-entrypoint.sh`) that maps SageMaker environment variables to TGI launcher settings.

If you are using Hugging Face’s SageMaker integration (recommended), you typically only need to set the model environment variables:

- **`HF_MODEL_ID`**: model id on the Hub (required)
- **`HF_MODEL_REVISION`**: optional revision
- **`SM_NUM_GPUS`**: number of GPUs (SageMaker sets this)
- **`HF_MODEL_QUANTIZE`**: optional quantization
- **`HF_MODEL_TRUST_REMOTE_CODE`**: optional trust remote code flag

For a minimal example using the Hugging Face SageMaker SDK and the official TGI image URI:

```python
import json
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

hub = {
    "HF_MODEL_ID": "HuggingFaceH4/zephyr-7b-beta",
    # SageMaker expects SM_NUM_GPUS to be a JSON-encoded int
    "SM_NUM_GPUS": json.dumps(1),
}

huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="3.3.5"),
    env=hub,
    role=role,
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
)

predictor.predict(
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is deep learning?"},
        ]
    }
)
```

## Benchmarking (what to measure, and how)

For meaningful benchmarks, measure both:

- **Client-visible latency** (end-to-end): p50/p95, time-to-first-token (TTFT), tokens/sec
- **Server-side performance metrics** (to attribute bottlenecks): see [metrics](../reference/metrics)

### End-to-end HTTP benchmark (recommended for EC2/SageMaker)

Use a load generator from *outside* the instance/endpoint VPC when possible (to include network overhead), and run a warmup phase before measuring.

Example approach:

1. Warm up with a small number of requests.
2. Run a fixed-duration load test at a target concurrency.
3. Record p50/p95 latency, error rate, and generated tokens/sec.

### Microbenchmark (model server only)

TGI also provides `text-generation-benchmark` (see the [benchmarking tool README](https://github.com/huggingface/text-generation-inference/tree/main/benchmark#readme)). This tool connects directly to the model server over a Unix socket and bypasses the router, so it’s useful for low-level profiling and batch-size sweeps, but it is **not** an end-to-end benchmark for SageMaker/HTTP.

