# Examples of Docker Commands for Gaudi Backend

This page gives a list of examples of docker run commands for some of the most popular models.

> **Note:** The parameters are chosen for Gaudi2 hardware to maximize performance on this given hardware, please adjust the parameters based on your hardware. For example, if you are using Gaudi3, you may want to increase the batch size.

## Default Precision (BF16)

### Llama3.1-8B on 1 card (BF16)

```bash
model=meta-llama/Meta-Llama-3.1-8B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -e HF_TOKEN=$hf_token \
   ghcr.io/huggingface/text-generation-inference:3.3.4-gaudi \
   --model-id $model \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-size 32 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

### Llama3.1-70B 8 cards (BF16)

```bash
model=meta-llama/Meta-Llama-3.1-70B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -e HF_TOKEN=$hf_token \
   ghcr.io/huggingface/text-generation-inference:3.3.4-gaudi \
   --model-id $model \
   --sharded true --num-shard 8 \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-size 256 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

### Llava-v1.6-Mistral-7B on 1 card (BF16)

```bash
model=llava-hf/llava-v1.6-mistral-7b-hf
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   ghcr.io/huggingface/text-generation-inference:3.3.4-gaudi \
   --model-id $model \
   --max-input-tokens 4096 --max-batch-prefill-tokens 16384 \
   --max-total-tokens 8192 --max-batch-size 4
```

## FP8 Precision

You could also set kv cache dtype to FP8 when launching the server, fp8_e4m3fn is supported in Gaudi

## Llama3-8B on 1 Card (FP8)

```bash
model=RedHatAI/Meta-Llama-3-8B-Instruct-FP8-KV
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -e HF_TOKEN=$hf_token \
   ghcr.io/huggingface/text-generation-inference:3.3.4-gaudi \
   --model-id $model \
   --kv-cache-dtype fp8_e4m3fn \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-size 32 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

## Llama3-70B on 8 cards (FP8)

```bash
model=RedHatAI/Meta-Llama-3-70B-Instruct-FP8
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -e HF_TOKEN=$hf_token \
   ghcr.io/huggingface/text-generation-inference:3.3.4-gaudi \
   --model-id $model \
   --kv-cache-dtype fp8_e4m3fn \
   --sharded true --num-shard 8 \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-size 256 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```
