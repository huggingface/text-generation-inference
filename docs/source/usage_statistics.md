
# Collection of Usage Statistics

Text Generation Inference collects anonymous usage statistics to help us improve the service. The collected data is used to improve TGI and to understand what causes failures. The data is collected transparently and any sensitive information is omitted.

Data is sent twice 1) on server startup and 2) and when server stops. Also, usage statistics are only enabled when TGI is running in docker to avoid collecting data then TGI runs directly on the host machine.

## What data is collected

The code that collects the data is available [here](https://github.com/huggingface/text-generation-inference/blob/main/router/src/usage_stats.rs).
As of release 2.1.2 this is an example of the data collected:

- From the TGI configuration:
```json
{
  "event_type": "start",           
  "disable_grammar_support": false,
  "json_output": false,
  "max_batch_prefill_tokens": 4096,
  "max_batch_size": null,
  "max_batch_total_tokens": null,
  "max_best_of": 2,
  "max_client_batch_size": 4,
  "max_concurrent_requests": 128,
  "max_input_tokens": 1024,
  "max_stop_sequences": 4,
  "max_top_n_tokens": 5,
  "max_total_tokens": 2048,
  "max_waiting_tokens": 20,
  "messages_api_enabled": false,
  "model_config": {
      "model_type": "bloom"
  },
  "ngrok": false,
  "revision": null,
  "tokenizer_class": "BloomTokenizerFast",
  "validation_workers": 2,
  "waiting_served_ratio": 1.2,
  "docker_label": "latest",
  "git_sha": "245d3de94877d4910ea7876f6bd19b4d7d4a47d9",
  "nvidia_env": "N/A",
  "system_env": {
      "architecture": "aarch64",
      "cpu_count": 16,
      "cpu_type": "Apple M3",
      "platform": "macos-unix-aarch64",
      "total_memory": 25769803767
  },
  "xpu_env": "N/A"
}
```

## How to opt-out

You can easily opt out by passing the `--disable-usage-stats` to the text-generation-launcher command. This will disable all usage statistics. You can also pass `--disable-crash-reports` which disables sending of crash reports, but allows anonymous usage statistics.
