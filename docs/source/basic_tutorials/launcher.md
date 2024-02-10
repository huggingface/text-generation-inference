# Text-generation-launcher arguments

<!-- WRAP CODE BLOCKS -->

```shell
Text Generation Launcher

Usage: text-generation-launcher [OPTIONS]

Options:
```
## MODEL_ID
```shell
      --model-id <MODEL_ID>
          The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers
          
          [env: MODEL_ID=]
          [default: bigscience/bloom-560m]

```
## REVISION
```shell
      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`
          
          [env: REVISION=]

```
## VALIDATION_WORKERS
```shell
      --validation-workers <VALIDATION_WORKERS>
          The number of tokenizer workers used for payload validation and truncation inside the router
          
          [env: VALIDATION_WORKERS=]
          [default: 2]

```
## SHARDED
```shell
      --sharded <SHARDED>
          Whether to shard the model across multiple GPUs By default text-generation-inference will use all available GPUs to run the model. Setting it to `false` deactivates `num_shard`
          
          [env: SHARDED=]
          [possible values: true, false]

```
## NUM_SHARD
```shell
      --num-shard <NUM_SHARD>
          The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance
          
          [env: NUM_SHARD=]

```
## QUANTIZE
```shell
      --quantize <QUANTIZE>
          Whether you want the model to be quantized
          
          [env: QUANTIZE=]

          Possible values:
          - awq:              4 bit quantization. Requires a specific AWQ quantized model: https://hf.co/models?search=awq. Should replace GPTQ models wherever possible because of the better latency
          - eetq:             8 bit quantization, doesn't require specific model. Should be a drop-in replacement to bitsandbytes with much better performance. Kernels are from https://github.com/NetEase-FuXi/EETQ.git
          - gptq:             4 bit quantization. Requires a specific GTPQ quantized model: https://hf.co/models?search=gptq. text-generation-inference will use exllama (faster) kernels wherever possible, and use triton kernel (wider support) when it's not. AWQ has faster kernels
          - bitsandbytes:     Bitsandbytes 8bit. Can be applied on any model, will cut the memory requirement in half, but it is known that the model will be much slower to run than the native f16
          - bitsandbytes-nf4: Bitsandbytes 4bit. Can be applied on any model, will cut the memory requirement by 4x, but it is known that the model will be much slower to run than the native f16
          - bitsandbytes-fp4: Bitsandbytes 4bit. nf4 should be preferred in most cases but maybe this one has better perplexity performance for you model

```
## SPECULATE
```shell
      --speculate <SPECULATE>
          The number of input_ids to speculate on If using a medusa model, the heads will be picked up automatically Other wise, it will use n-gram speculation which is relatively free in terms of compute, but the speedup heavily depends on the task
          
          [env: SPECULATE=]

```
## DTYPE
```shell
      --dtype <DTYPE>
          The dtype to be forced upon the model. This option cannot be used with `--quantize`
          
          [env: DTYPE=]
          [possible values: float16, bfloat16]

```
## TRUST_REMOTE_CODE
```shell
      --trust-remote-code
          Whether you want to execute hub modelling code. Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision
          
          [env: TRUST_REMOTE_CODE=]

```
## MAX_CONCURRENT_REQUESTS
```shell
      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly
          
          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 128]

```
## MAX_BEST_OF
```shell
      --max-best-of <MAX_BEST_OF>
          This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time, and return the best in terms of overall log probability over the entire generated sequence
          
          [env: MAX_BEST_OF=]
          [default: 2]

```
## MAX_STOP_SEQUENCES
```shell
      --max-stop-sequences <MAX_STOP_SEQUENCES>
          This is the maximum allowed value for clients to set `stop_sequences`. Stop sequences are used to allow the model to stop on more than just the EOS token, and enable more complex "prompting" where users can preprompt the model in a specific way and define their "own" stop token aligned with their prompt
          
          [env: MAX_STOP_SEQUENCES=]
          [default: 4]

```
## MAX_TOP_N_TOKENS
```shell
      --max-top-n-tokens <MAX_TOP_N_TOKENS>
          This is the maximum allowed value for clients to set `top_n_tokens`. `top_n_tokens is used to return information about the the `n` most likely tokens at each generation step, instead of just the sampled token. This information can be used for downstream tasks like for classification or ranking
          
          [env: MAX_TOP_N_TOKENS=]
          [default: 5]

```
## MAX_INPUT_LENGTH
```shell
      --max-input-length <MAX_INPUT_LENGTH>
          This is the maximum allowed input length (expressed in number of tokens) for users. The larger this value, the longer prompt users can send which can impact the overall memory required to handle the load. Please note that some models have a finite range of sequence they can handle
          
          [env: MAX_INPUT_LENGTH=]
          [default: 1024]

```
## MAX_TOTAL_TOKENS
```shell
      --max-total-tokens <MAX_TOTAL_TOKENS>
          This is the most important value to set as it defines the "memory budget" of running clients requests. Clients will send input sequences and ask to generate `max_new_tokens` on top. with a value of `1512` users can send either a prompt of `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for `1511` max_new_tokens. The larger this value, the larger amount each request will be in your RAM and the less effective batching can be
          
          [env: MAX_TOTAL_TOKENS=]
          [default: 2048]

```
## WAITING_SERVED_RATIO
```shell
      --waiting-served-ratio <WAITING_SERVED_RATIO>
          This represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch. `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's only 10 queries left in the current batch we check if we can fit those 12 waiting queries into the batching strategy, and if yes, then batching happens delaying the 10 running queries by a `prefill` run.
          
          This setting is only applied if there is room in the batch as defined by `max_batch_total_tokens`.
          
          [env: WAITING_SERVED_RATIO=]
          [default: 1.2]

```
## MAX_BATCH_PREFILL_TOKENS
```shell
      --max-batch-prefill-tokens <MAX_BATCH_PREFILL_TOKENS>
          Limits the number of tokens for the prefill operation. Since this operation take the most memory and is compute bound, it is interesting to limit the number of requests that can be sent
          
          [env: MAX_BATCH_PREFILL_TOKENS=]
          [default: 4096]

```
## MAX_BATCH_TOTAL_TOKENS
```shell
      --max-batch-total-tokens <MAX_BATCH_TOTAL_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.
          
          This represents the total amount of potential tokens within a batch. When using padding (not recommended) this would be equivalent of `batch_size` * `max_total_tokens`.
          
          However in the non-padded (flash attention) version this can be much finer.
          
          For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.
          
          Overall this number should be the largest possible amount that fits the remaining memory (after the model is loaded). Since the actual memory overhead depends on other parameters like if you're using quantization, flash attention or the model implementation, text-generation-inference cannot infer this number automatically.
          
          [env: MAX_BATCH_TOTAL_TOKENS=]

```
## MAX_WAITING_TOKENS
```shell
      --max-waiting-tokens <MAX_WAITING_TOKENS>
          This setting defines how many tokens can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). New queries require 1 `prefill` forward, which is different from `decode` and therefore you need to pause the running batch in order to run `prefill` to create the correct values for the waiting queries to be able to join the batch.
          
          With a value too small, queries will always "steal" the compute to run `prefill` and running queries will be delayed by a lot.
          
          With a value too big, waiting queries could wait for a very long time before being allowed a slot in the running batch. If your server is busy that means that requests that could run in ~2s on an empty server could end up running in ~20s because the query had to wait for 18s.
          
          This number is expressed in number of tokens to make it a bit more "model" agnostic, but what should really matter is the overall latency for end users.
          
          [env: MAX_WAITING_TOKENS=]
          [default: 20]

```
## MAX_BATCH_SIZE
```shell
      --max-batch-size <MAX_BATCH_SIZE>
          Enforce a maximum number of requests per batch Specific flag for hardware targets that do not support unpadded inference
          
          [env: MAX_BATCH_SIZE=]

```
## ENABLE_CUDA_GRAPHS
```shell
      --enable-cuda-graphs
          Enable experimental support for cuda graphs
          
          [env: ENABLE_CUDA_GRAPHS=]

```
## HOSTNAME
```shell
      --hostname <HOSTNAME>
          The IP address to listen on
          
          [env: HOSTNAME=]
          [default: 0.0.0.0]

```
## PORT
```shell
  -p, --port <PORT>
          The port to listen on
          
          [env: PORT=]
          [default: 3000]

```
## SHARD_UDS_PATH
```shell
      --shard-uds-path <SHARD_UDS_PATH>
          The name of the socket for gRPC communication between the webserver and the shards
          
          [env: SHARD_UDS_PATH=]
          [default: /tmp/text-generation-server]

```
## MASTER_ADDR
```shell
      --master-addr <MASTER_ADDR>
          The address the master shard will listen on. (setting used by torch distributed)
          
          [env: MASTER_ADDR=]
          [default: localhost]

```
## MASTER_PORT
```shell
      --master-port <MASTER_PORT>
          The address the master port will listen on. (setting used by torch distributed)
          
          [env: MASTER_PORT=]
          [default: 29500]

```
## HUGGINGFACE_HUB_CACHE
```shell
      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance
          
          [env: HUGGINGFACE_HUB_CACHE=]

```
## WEIGHTS_CACHE_OVERRIDE
```shell
      --weights-cache-override <WEIGHTS_CACHE_OVERRIDE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance
          
          [env: WEIGHTS_CACHE_OVERRIDE=]

```
## DISABLE_CUSTOM_KERNELS
```shell
      --disable-custom-kernels
          For some models (like bloom), text-generation-inference implemented custom cuda kernels to speed up inference. Those kernels were only tested on A100. Use this flag to disable them if you're running on different hardware and encounter issues
          
          [env: DISABLE_CUSTOM_KERNELS=]

```
## CUDA_MEMORY_FRACTION
```shell
      --cuda-memory-fraction <CUDA_MEMORY_FRACTION>
          Limit the CUDA available memory. The allowed value equals the total visible memory multiplied by cuda-memory-fraction
          
          [env: CUDA_MEMORY_FRACTION=]
          [default: 1.0]

```
## ROPE_SCALING
```shell
      --rope-scaling <ROPE_SCALING>
          Rope scaling will only be used for RoPE models and allow rescaling the position rotary to accomodate for larger prompts.
          
          Goes together with `rope_factor`.
          
          `--rope-factor 2.0` gives linear scaling with a factor of 2.0 `--rope-scaling dynamic` gives dynamic scaling with a factor of 1.0 `--rope-scaling linear` gives linear scaling with a factor of 1.0 (Nothing will be changed basically)
          
          `--rope-scaling linear --rope-factor` fully describes the scaling you want
          
          [env: ROPE_SCALING=]
          [possible values: linear, dynamic]

```
## ROPE_FACTOR
```shell
      --rope-factor <ROPE_FACTOR>
          Rope scaling will only be used for RoPE models See `rope_scaling`
          
          [env: ROPE_FACTOR=]

```
## JSON_OUTPUT
```shell
      --json-output
          Outputs the logs in JSON format (useful for telemetry)
          
          [env: JSON_OUTPUT=]

```
## OTLP_ENDPOINT
```shell
      --otlp-endpoint <OTLP_ENDPOINT>
          [env: OTLP_ENDPOINT=]

```
## CORS_ALLOW_ORIGIN
```shell
      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          [env: CORS_ALLOW_ORIGIN=]

```
## WATERMARK_GAMMA
```shell
      --watermark-gamma <WATERMARK_GAMMA>
          [env: WATERMARK_GAMMA=]

```
## WATERMARK_DELTA
```shell
      --watermark-delta <WATERMARK_DELTA>
          [env: WATERMARK_DELTA=]

```
## NGROK
```shell
      --ngrok
          Enable ngrok tunneling
          
          [env: NGROK=]

```
## NGROK_AUTHTOKEN
```shell
      --ngrok-authtoken <NGROK_AUTHTOKEN>
          ngrok authentication token
          
          [env: NGROK_AUTHTOKEN=]

```
## NGROK_EDGE
```shell
      --ngrok-edge <NGROK_EDGE>
          ngrok edge
          
          [env: NGROK_EDGE=]

```
## TOKENIZER_CONFIG_PATH
```shell
      --tokenizer-config-path <TOKENIZER_CONFIG_PATH>
          The path to the tokenizer config file. This path is used to load the tokenizer configuration which may include a `chat_template`. If not provided, the default config will be used from the model hub
          
          [env: TOKENIZER_CONFIG_PATH=]

```
## ENV
```shell
  -e, --env
          Display a lot of information about your runtime environment

```
## HELP
```shell
  -h, --help
          Print help (see a summary with '-h')

```
## VERSION
```shell
  -V, --version
          Print version

```
