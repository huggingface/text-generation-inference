<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Text Generation Inference on Habana Gaudi

## Table of contents

- [Running TGI on Gaudi](#running-tgi-on-gaudi)
- [Adjusting TGI parameters](#adjusting-tgi-parameters)
- [Running TGI with FP8 precision](#running-tgi-with-fp8-precision)
- [Currently supported configurations](#currently-supported-configurations)
- [Environment variables](#environment-variables)
- [Profiler](#profiler)

## Running TGI on Gaudi

To use [🤗 text-generation-inference](https://github.com/huggingface/text-generation-inference) on Habana Gaudi/Gaudi2, follow these steps:

1. Pull the official Docker image with:
   ```bash
   docker pull ghcr.io/huggingface/tgi-gaudi:2.0.0
   ```
> [!NOTE]
> Alternatively, you can build the Docker image using the `Dockerfile` located in this folder with:
> ```bash
> docker build -t tgi_gaudi .
> ```
2. Launch a local server instance:

    i. On 1 Gaudi/Gaudi2 card
   ```bash
   model=meta-llama/Llama-2-7b-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.0 --model-id $model --max-input-length 1024 --max-total-tokens 2048
   ```
   > For gated models such as [LLama](https://huggingface.co/meta-llama) or [StarCoder](https://huggingface.co/bigcode/starcoder), you will have to pass `-e HUGGING_FACE_HUB_TOKEN=<token>` to the `docker run` command above with a valid Hugging Face Hub read token.

    ii. On 8 Gaudi/Gaudi2 cards:
   ```bash
   model=meta-llama/Llama-2-70b-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.0 --model-id $model --sharded true --num-shard 8 --max-input-length 1024 --max-total-tokens 2048
   ```
3. You can then send a simple request:
   ```bash
   curl 127.0.0.1:8080/generate \
     -X POST \
     -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32}}' \
     -H 'Content-Type: application/json'
   ```
4. To run static benchmark test, please refer to [TGI's benchmark tool](https://github.com/huggingface/text-generation-inference/tree/main/benchmark).

   To run it on the same machine, you can do the following:
   * `docker exec -it <docker name> bash` , pick the docker started from step 2 using docker ps
   * `text-generation-benchmark -t <model-id>` , pass the model-id from docker run command
   * after the completion of tests, hit ctrl+c to see the performance data summary.

5. To run continuous batching test, please refer to [examples](https://github.com/huggingface/tgi-gaudi/tree/habana-main/examples).

## Adjusting TGI parameters

Maximum sequence length is controlled by two arguments:
- `--max-input-length` is the maximum possible input prompt length. Default value is `4095`.
- `--max-total-tokens` is the maximum possible total length of the sequence (input and output). Default value is `4096`.

Maximum batch size is controlled by two arguments:
- For prefill operation, please set `--max-prefill-total-tokens` as `bs * max-input-length`, where `bs` is your expected maximum prefill batch size.
- For decode operation, please set `--max-batch-total-tokens` as `bs * max-total-tokens`, where `bs` is your expected maximum decode batch size.
- Please note that batch size will be always padded to the nearest multiplication of `BATCH_BUCKET_SIZE` and `PREFILL_BATCH_BUCKET_SIZE`.

To ensure greatest performance results, at the begginging of each server run, warmup is performed. It's designed to cover major recompilations while using HPU Graphs. It creates queries with all possible input shapes, based on provided parameters (described in this section) and runs basic TGI operations on them (prefill, decode, concatenate).

Except those already mentioned, there are other parameters that need to be properly adjusted to improve performance or memory usage:

- `PAD_SEQUENCE_TO_MULTIPLE_OF` determines sizes of input legnth buckets. Since warmup creates several graphs for each bucket, it's important to adjust that value proportionally to input sequence length. Otherwise, some out of memory issues can be observed.
- `ENABLE_HPU_GRAPH` enables HPU graphs usage, which is crucial for performance results. Recommended value to keep is `true` .

For more information and documentation about Text Generation Inference, checkout [the README](https://github.com/huggingface/text-generation-inference#text-generation-inference) of the original repo.

## Running TGI with FP8 precision

TGI supports FP8 precision runs within the limits provided by [Habana Quantization Toolkit](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html). Models with FP8 can be ran by properly setting QUANT_CONFIG environment variable. Detailed instruction on how to use that variable can be found in [Optimum Habana FP8 guide](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#running-with-fp8). Summarising that instruction in TGI cases:

1. Measure quantization statistics of requested model by using [Optimum Habana measurement script](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#running-with-fp8:~:text=use_deepspeed%20%2D%2Dworld_size%208-,run_lm_eval.py,-%5C%0A%2Do%20acc_70b_bs1_measure.txt)
2. Run requested model in TGI with proper QUANT_CONFIG setting - e.g. `QUANT_CONFIG=./quantization_config/maxabs_quant.json`

> [!NOTE]
> Only models pointed in [supported configurations](#currently-supported-configurations) are guaranteed to work with FP8

Additional hints to quantize model for TGI when using `run_lm_eval.py`:
* use `--limit_hpu_graphs` flag to save memory
* try to model your use case situation by adjusting `--batch_size` , `--max_new_tokens 512` and `--max_input_tokens 512`; in case of memory issues, lower those values
* use dataset/tasks suitable for your use case (see `--help` for defining tasks/datasets)

## Currently supported configurations

Not all features of TGI are currently supported as this is still a work in progress.
Currently supported and validated configurations (other configurations are not guaranted to work or ensure reasonable performance):

<div align="left">

| Model| Cards| Decode batch size| Dtype| Max input tokens |Max total tokens|
|:----:|:----:|:----------------:|:----:|:----------------:|:--------------:|
| LLaMA 70b | 8     | 128 | bfloat16/FP8 | 1024 | 2048 |
| LLaMA 7b  | 1/8   | 16  | bfloat16/FP8 | 1024 | 2048 |
</div>

Other sequence lengths can be used with proportionally decreased/increased batch size (the higher sequence length, the lower batch size).
Support for other models from Optimum Habana will be added successively.

## Environment variables

<div align="left">

| Name                        | Value(s)   | Default          | Description                                                                                                                      | Usage                        |
| --------------------------- | :--------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| ENABLE_HPU_GRAPH            | True/False | True             | Enable hpu graph or not                                                                                                          | add -e in docker run command |
| LIMIT_HPU_GRAPH             | True/False | False            | Skip HPU graph usage for prefill to save memory, set to `True` for large sequence/decoding lengths(e.g. 300/212)                 | add -e in docker run command |
| BATCH_BUCKET_SIZE           | integer    | 8                | Batch size for decode operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs  | add -e in docker run command |
| PREFILL_BATCH_BUCKET_SIZE   | integer    | 4                | Batch size for prefill operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs | add -e in docker run command |
| PAD_SEQUENCE_TO_MULTIPLE_OF | integer    | 128              | For prefill operation, sequences will be padded to a multiple of provided value.                                                 | add -e in docker run command |
| SKIP_TOKENIZER_IN_TGI       | True/False | False            | Skip tokenizer for input/output processing                                                                                       | add -e in docker run command |
| WARMUP_ENABLED              | True/False | True             | Enable warmup during server initialization to recompile all graphs. This can increase TGI setup time.                            | add -e in docker run command |
| QUEUE_THRESHOLD_MS          | integer    | 120              | Controls the threshold beyond which the request are considered overdue and handled with priority. Shorter requests are prioritized otherwise.                            | add -e in docker run command |
</div>

## Profiler

To collect performance profiling, please set below environment variables:

<div align="left">

| Name               | Value(s)   | Default          | Description                                              | Usage                        |
| ------------------ | :--------- | :--------------- | :------------------------------------------------------- | :--------------------------- |
| PROF_WAITSTEP      | integer    | 0                | Control profile wait steps                               | add -e in docker run command |
| PROF_WARMUPSTEP    | integer    | 0                | Control profile warmup steps                             | add -e in docker run command |
| PROF_STEP          | integer    | 0                | Enable/disable profile, control profile active steps     | add -e in docker run command |
| PROF_PATH          | string     | /tmp/hpu_profile | Define profile folder                                    | add -e in docker run command |
| PROF_RANKS         | string     | 0                | Comma-separated list of ranks to profile                 | add -e in docker run command |
| PROF_RECORD_SHAPES | True/False | False            | Control record_shapes option in the profiler             | add -e in docker run command |
</div>



> The license to use TGI on Habana Gaudi is the one of TGI: https://github.com/huggingface/text-generation-inference/blob/main/LICENSE
>
> Please reach out to api-enterprise@huggingface.co if you have any question.
