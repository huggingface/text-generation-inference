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

To use [🤗 text-generation-inference](https://github.com/huggingface/text-generation-inference) on Habana Gaudi/Gaudi2, follow these steps:

1. Build the Docker image located in this folder with:
   ```bash
   docker build -t tgi_gaudi .
   ```
2. Launch a local server instance on 1 Gaudi card:
   ```bash
   model=meta-llama/Llama-2-7b-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model
   ```
3. Launch a local server instance on 8 Gaudi cards:
   ```bash
   model=meta-llama/Llama-2-70b-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model --sharded true --num-shard 8
   ```
   > Set `LIMIT_HPU_GRAPH=True` for larger sequence/decoding lengths(e.g. 300/212).
4. You can then send a request:
   ```bash
   curl 127.0.0.1:8080/generate \
     -X POST \
     -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
     -H 'Content-Type: application/json'
   ```
   > The first call will be slower as the model is compiled.
5. To run benchmark test, please refer [TGI's benchmark tool](https://github.com/huggingface/text-generation-inference/tree/main/benchmark).

   To run it on the same machine, you can do the following:
   * `docker exec -it <docker name> bash` , pick the docker started from step 3 or 4 using docker ps
   * `text-generation-benchmark -t <model-id>` , pass the model-id from docker run command
   * after the completion of tests, hit ctrl+c to see the performance data summary.

> For gated models such as [StarCoder](https://huggingface.co/bigcode/starcoder), you will have to pass `-e HUGGING_FACE_HUB_TOKEN=<token>` to the `docker run` command above with a valid Hugging Face Hub read token.

For more information and documentation about Text Generation Inference, checkout [the README](https://github.com/huggingface/text-generation-inference#text-generation-inference) of the original repo.

Not all features of TGI are currently supported as this is still a work in progress.

New changes are added for the current release:
- Sharded feature with support for DeepSpeed-inference auto tensor parallelism. Also, use HPU graphs for performance improvement.
- Torch profile.
- Batch size bucketing for decode and prefill.



Environment Variables Added:

<div align="center">

| Name                  | Value(s)       | Default     | Description                       | Usage                                          |
|------------------     |:---------------|:------------|:--------------------              |:---------------------------------
|  MAX_TOTAL_TOKENS     | integer        | 0           | Control the padding of input          | add -e in docker run, such         |
|  ENABLE_HPU_GRAPH     | true/false     | true        | Enable hpu graph or not                                                      |  add -e in docker run command  |
|  PROF_WARMUPSTEP      | integer        | 0           | Enable/disable profile, control profile warmup step, 0 means disable profile |  add -e in docker run command  |
|  PROF_STEP            | interger       | 5           | Control profile step                                                         |  add -e in docker run command  |
|  PROF_PATH            | string         | /root/text-generation-inference                                   | Define profile folder  | add -e in docker run command  |
| LIMIT_HPU_GRAPH       | True/False     | True       | Skip HPU graph usage for prefill to save memory, set to `True` for large sequence/decoding lengths(e.g. 300/212) | add -e in docker run command |
| BATCH_BUCKET_SIZE     | integer        | 8           | Batch size for decode operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs | add -e in docker run command |
| PREFILL_BATCH_BUCKET_SIZE     | integer        | 4           | Batch size for prefill operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs | add -e in docker run command |
| SKIP_TOKENIZER_IN_TGI | True/False     | False        | Skip tokenizer for input/output processing | add -e in docker run command |
|  TGI_PROFILER_ENABLED | True/False     | False       | Collect high-level server tracing events          | add -e in docker run command  |

</div>


Maximum batch size is controlled by two arguments:
- For prefill operation, please set `--max-prefill-total-tokens` as `bs * max-input-length`, where `bs` is your expected maximum prefill batch size.
- For decode operation, please set `--max-batch-total-tokens` as `bs * max-total-tokens`, where `bs` is your expected maximum decode batch size.
- Please note that batch size will be always padded to the nearest multiplication of `BATCH_BUCKET_SIZE` and `PREFILL_BATCH_BUCKET_SIZE`.

Current limitations:
- `LIMIT_HPU_GRAPH=False` causes accuracy issues and it should be avoided.
- Memory usage is higher than expected. Please consider using smaller batch sizes.

> The license to use TGI on Habana Gaudi is the one of TGI: https://github.com/huggingface/text-generation-inference/blob/main/LICENSE
>
> Please reach out to api-enterprise@huggingface.co if you have any question.
