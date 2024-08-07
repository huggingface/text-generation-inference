
# Collection of Usage Statistics

Text Generation Inference collects anonymous usage statistics to help us improve the service. The collected data is used to improve TGI and to understand what causes failures. The data is collected transparently and any sensitive information is omitted.

Data is sent twice, once on server startup and once when server stops. Also, usage statistics are only enabled when TGI is running in docker to avoid collecting data then TGI runs directly on the host machine.

## What data is collected

The code that collects the data is available [here](https://github.com/huggingface/text-generation-inference/blob/main/router/src/usage_stats.rs).
As of release 2.1.2 this is an example of the data collected:

- From the TGI configuration:
```json
{
  "event_type": "start",
  "disable_grammar_support": false,
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
    "model_type": "Bloom"
  },
  "revision": null,
  "tokenizer_class": "BloomTokenizerFast",
  "validation_workers": 2,
  "waiting_served_ratio": 1.2,
  "docker_label": "latest",
  "git_sha": "cfc118704880453d29bcbe4fbbd91dda501cf5fe",
  "nvidia_env": {
    "name": "NVIDIA A10G",
    "pci_bus_id": "00000000:00:1E.0",
    "driver_version": "535.183.01",
    "pstate": "P8",
    "pcie_link_gen_max": "4",
    "pcie_link_gen_current": "1",
    "temperature_gpu": "31",
    "utilization_gpu": "0 %",
    "utilization_memory": "0 %",
    "memory_total": "23028 MiB",
    "memory_free": "22515 MiB",
    "memory_used": "0 MiB",
    "reset_status_reset_required": "No",
    "reset_status_drain_and_reset_recommended": "No",
    "compute_cap": "8.6",
    "ecc_errors_corrected_volatile_total": "0",
    "mig_mode_current": "[N/A]",
    "power_draw_instant": "10.86 W",
    "power_limit": "300.00 W"
  },
  "system_env": {
    "cpu_count": 16,
    "cpu_type": "AMD EPYC 7R32",
    "total_memory": 66681196544,
    "architecture": "x86_64",
    "platform": "linux-unix-x86_64"
  }
}

```

## How to opt-out

By passing the `--usage-stats` to the text-generation-launcher you can control how much usage statistics are being collected.
`--usage-stats=no-stack` will not emit the stack traces from errors and the error types, but will continue to send start and stop events
`--usage-stats=off` will completely disable everything
