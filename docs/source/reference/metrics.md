# Metrics

TGI exposes multiple metrics that can be collected via the `/metrics` Prometheus endpoint.
These metrics can be used to monitor the performance of TGI, autoscale deployment and to help identify bottlenecks.

The following metrics are exposed:

| Metric Name                                | Description                                                                              | Type      | Unit    |
|--------------------------------------------|------------------------------------------------------------------------------------------|-----------|---------|
| `tgi_batch_current_max_tokens`             | Maximum tokens for the current batch                                                     | Gauge     | Count   |
| `tgi_batch_current_size`                   | Current batch size                                                                       | Gauge     | Count   |
| `tgi_batch_decode_duration`                | Time spent decoding a batch per method (prefill or decode)                               | Histogram | Seconds |
| `tgi_batch_filter_duration`                | Time spent filtering batches and sending generated tokens per method (prefill or decode) | Histogram | Seconds |
| `tgi_batch_forward_duration`               | Batch forward duration per method (prefill or decode)                                    | Histogram | Seconds |
| `tgi_batch_inference_count`                | Inference calls per method (prefill or decode)                                           | Counter   | Count   |
| `tgi_batch_inference_duration`             | Batch inference duration                                                                 | Histogram | Seconds |
| `tgi_batch_inference_success`              | Number of successful inference calls per method (prefill or decode)                      | Counter   | Count   |
| `tgi_batch_next_size`                      | Batch size of the next batch                                                             | Histogram | Count   |
| `tgi_queue_size`                           | Current queue size                                                                       | Gauge     | Count   |
| `tgi_request_count`                        | Total number of requests                                                                 | Counter   | Count   |
| `tgi_request_duration`                     | Total time spent processing the request (e2e latency)                                    | Histogram | Seconds |
| `tgi_request_generated_tokens`             | Generated tokens per request                                                             | Histogram | Count   |
| `tgi_request_inference_duration`           | Request inference duration                                                               | Histogram | Seconds |
| `tgi_request_input_length`                 | Input token length per request                                                           | Histogram | Count   |
| `tgi_request_max_new_tokens`               | Maximum new tokens per request                                                           | Histogram | Count   |
| `tgi_request_mean_time_per_token_duration` | Mean time per token per request (inter-token latency)                                    | Histogram | Seconds |
| `tgi_request_queue_duration`               | Time spent in the queue per request                                                      | Histogram | Seconds |
| `tgi_request_skipped_tokens`               | Speculated tokens per request                                                            | Histogram | Count   |
| `tgi_request_success`                      | Number of successful requests                                                            | Counter   |         |
| `tgi_request_validation_duration`          | Time spent validating the request                                                        | Histogram | Seconds |
