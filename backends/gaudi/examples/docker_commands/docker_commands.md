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
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
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
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --sharded true --num-shard 8 \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-size 256 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

### Llama2-7B on 1 Card (BF16)

```bash
model=meta-llama/Llama-2-7b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -e HF_TOKEN=$hf_token \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-size 32 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

### Llama2-70B on 8 cards (BF16)

```bash
model=meta-llama/Llama-2-70b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -e HF_TOKEN=$hf_token \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
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
    -e PREFILL_BATCH_BUCKET_SIZE=1 \
    -e BATCH_BUCKET_SIZE=1 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --max-input-tokens 4096 --max-batch-prefill-tokens 16384 \
   --max-total-tokens 8192 --max-batch-size 4
```

## FP8 Precision

Please refer to the [FP8 Precision](https://huggingface.co/docs/text-generation-inference/backends/gaudi_new#how-to-use-different-precision-formats) section for more details. You need to measure the statistics of the model first before running the model in FP8 precision.

## Llama3.1-8B on 1 Card (FP8)

```bash
model=meta-llama/Meta-Llama-3.1-8B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HF_TOKEN=$hf_token \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-size 32 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

## Llama3.1-70B on 8 cards (FP8)

```bash
model=meta-llama/Meta-Llama-3.1-70B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HF_TOKEN=$hf_token \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --sharded true --num-shard 8 \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-size 256 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

## Llama2-7B on 1 Card (FP8)

```bash
model=meta-llama/Llama-2-7b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HF_TOKEN=$hf_token \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-size 32 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

## Llama2-70B on 8 Cards (FP8)

```bash
model=meta-llama/Llama-2-70b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HF_TOKEN=$hf_token \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --sharded true --num-shard 8 \
   --max-input-tokens 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-size 256 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

## Llava-v1.6-Mistral-7B on 1 Card (FP8)

```bash
model=llava-hf/llava-v1.6-mistral-7b-hf
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
    -e PREFILL_BATCH_BUCKET_SIZE=1 \
    -e BATCH_BUCKET_SIZE=1 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --max-input-tokens 4096 --max-batch-prefill-tokens 16384 \
   --max-total-tokens 8192 --max-batch-size 4
```

## Llava-v1.6-Mistral-7B on 8 Cards (FP8)

```bash
model=llava-hf/llava-v1.6-mistral-7b-hf
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   --cap-add=sys_nice \
   --ipc=host \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
    -e PREFILL_BATCH_BUCKET_SIZE=1 \
    -e BATCH_BUCKET_SIZE=1 \
   ghcr.io/huggingface/text-generation-inference:3.1.1-gaudi \
   --model-id $model \
   --sharded true --num-shard 8 \
   --max-input-tokens 4096 --max-batch-prefill-tokens 16384 \
   --max-total-tokens 8192 --max-batch-size 4
```
