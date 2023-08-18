# Configuration parameters for Text Generation Inference

Text Generation Inference allows you to customize the way you serve your models. You can use the following parameters to configure your server. You can enable them by adding them environment variables or by providing them as arguments when running `text-generation-launcher`. Environment variables are in `UPPER_CASE` and arguments are in `lower_case`.

## Model

- `MODEL_ID` - The name of the model to load. Can be a MODEL_ID as listed on huggingface.co like `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers. Default: `bigscience/bloom-560m`.

- `REVISION` - The actual revision of the model if you're referring to a model on Hugging Face Hub. You can use a specific commit id or a branch like `refs/pr/2`.

- `QUANTIZE` - Whether you want the model to be quantized. This will use `bitsandbytes` for quantization on the fly, or `gptq`. 4bit quantization is available through `bitsandbytes` by providing the `bitsandbytes-fp4` or `bitsandbytes-nf4` options. 

- `DTYPE` - The dtype to be forced upon the model. This option cannot be used with `--quantize`.

- `TRUST_REMOTE_CODE` - Whether you want to execute Hub modelling code. Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision. Default: false

- `DISABLE_CUSTOM_KERNELS` - For some models (like bloom), text-generation-inference implemented custom cuda kernels to speed up inference. Those kernels were only tested on A100. Use this flag to disable them if you're running on different hardware and encounter issues. Default: false

- `ROPE_SCALING` - Rope scaling will only be used for RoPE models and allow rescaling the position rotary to accomodate for larger prompts. Goes together with `rope_factor`. Default: linear

- `ROPE_FACTOR` - Rope scaling factor. Default: 1.0

## Inference Settings

- `VALIDATION_WORKERS` - The number of tokenizer workers used for payload validation and truncation inside the router. Default: 2

- `SHARDED` - Whether to shard the model across multiple GPUs. By default text-generation-inference will use all available GPUs to run the model. Setting it to `false` deactivates `num_shard`. 

- `NUM_SHARD` - The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance.

- `MAX_CONCURRENT_REQUESTS` - The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly. Default: 128

- `MAX_BEST_OF` - This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time, and return the best in terms of overall log probability over the entire generated sequence. Default: 2

- `MAX_STOP_SEQUENCES` - This is the maximum allowed value for clients to set `stop_sequences`. Stop sequences are used to allow the model to stop on more than just the EOS token, and enable more complex "prompting" where users can preprompt the model in a specific way and define their "own" stop token aligned with their prompt. Default: 4

- `MAX_INPUT_LENGTH` - This is the maximum allowed input length (expressed in number of tokens) for users. The larger this value, the longer prompt users can send which can impact the overall memory required to handle the load. Please note that some models have a finite range of sequence they can handle. Default: 1024

- `MAX_TOTAL_TOKENS` - This is the most important value to set as it defines the "memory budget" of running clients requests. Clients will send input sequences and ask to generate `max_new_tokens` on top. with a value of `1512` users can send either a prompt of `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for `1511` max_new_tokens. The larger this value, the larger amount each request will be in your RAM and the less effective batching can be. Default: 2048

- `WAITING_SERVED_RATIO` - This represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch. `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's only 10 queries left in the current batch we check if we can fit those 12 waiting queries into the batching strategy, and if yes, then batching happens delaying the 10 running queries by a `prefill` run. This setting is only applied if there is room in the batch as defined by `max_batch_total_tokens`. Default: 1.2

- `MAX_BATCH_PREFILL_TOKENS` - Limits the number of tokens for the prefill operation. Since this operation take the most memory and is compute bound, it is interesting to limit the number of requests that can be sent. Default: 4096

- `MAX_BATCH_TOTAL_TOKENS` - **IMPORTANT** This is one critical control to allow maximum usage of the available hardware. This represents the total amount of potential tokens within a batch. When using padding (not recommended) this would be equivalent of `batch_size` * `max_total_tokens`. However in the non-padded (flash attention) version this can be much finer. For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens. Overall this number should be the largest possible amount that fits the remaining memory (after the model is loaded). Since the actual memory overhead depends on other parameters like if you're using quantization, flash attention or the model implementation, text-generation-inference cannot infer this number automatically.

- `MAX_WAITING_TOKENS` - This setting defines how many tokens can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). New queries require 1 `prefill` forward, which is different from `decode` and therefore you need to pause the running batch in order to run `prefill` to create the correct values for the waiting queries to be able to join the batch. With a value too small, queries will always "steal" the compute to run `prefill` and running queries will be delayed by a lot. With a value too big, waiting queries could wait for a very long time before being allowed a slot in the running batch. Default: 20

## Server 

- `HOSTNAME` - The IP address to listen on. Default: `0.0.0.0`

- `PORT` - The port to listen on. Default: 3000

- `SHARD_UDS_PATH` - The name of the socket for gRPC communication between the webserver and the shards. Default: `/tmp/text-generation-server`

- `MASTER_ADDR` - The address the master shard will listen on. (setting used by torch distributed). Default: `localhost` 

- `MASTER_PORT` - The address the master port will listen on. (setting used by torch distributed). Default: 29500

- `HUGGINGFACE_HUB_CACHE` - The location of the Hugging Face Hub cache. Used to override the location if you want to provide a mounted disk for instance.

- `WEIGHTS_CACHE_OVERRIDE` - The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance.

- `HUGGING_FACE_HUB_TOKEN` - The token to use to authenticate to the Hugging Face Hub. Used to download private models.


## Logging

- `JSON_OUTPUT` - Outputs the logs in JSON format (useful for telemetry). Default: false

- `OTLP_ENDPOINT` - Send metrics to the OpenTelemetry endpoint. 

- `CORS_ALLOW_ORIGIN` - Allowed CORS origins.