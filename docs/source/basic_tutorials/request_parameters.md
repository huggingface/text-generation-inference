# Request Parameters for Text Generation Inference

Text Generation Inference supports different parameters to control the generation, defining them in the `parameters`` attribute of the payload. 

```bash
curl https://j4xhm53fxl9ussm8.us-east-1.aws.endpoints.huggingface.cloud \
-X POST \
-d '{"inputs":"Once upon a time,", "parameters": {"max_new_tokens": 256}}' \
-H "Authorization: Bearer <hf_token>" \
-H "Content-Type: application/json"
```


As of today, the following parameters are supported:

- `temperature`: Controls randomness in the model. Lower values will make the model more deterministic and higher values will make the model more random. Default value is 1.0.
- `max_new_tokens`: The maximum number of tokens to generate. Default value is 20, max value is 512.
- `repetition_penalty`: Controls the likelihood of repetition. Default is `null`.
- `seed`: The seed to use for random generation. Default is `null`.
- `stop`: A list of tokens to stop the generation. The generation will stop when one of the tokens is generated.
- `top_k`: The number of highest probability vocabulary tokens to keep for top-k-filtering. Default value is `null`, which disables top-k-filtering.
- `top_p`: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling, default to `null`
- `do_sample`: Whether or not to use sampling; use greedy decoding otherwise. Default value is `false`.
- `best_of`: Generate best_of sequences and return the one if the highest token logprobs, default to `null`.
- `details`: Whether or not to return details about the generation. Default value is `false`.
- `return_full_text`: Whether or not to return the full text or only the generated part. Default value is `false`.
- `truncate`: Whether or not to truncate the input to the maximum length of the model. Default value is `true`.
- `typical_p`: The typical probability of a token. Default value is `null`.
- `watermark`: The watermark to use for the generation. Default value is `false`.