# Non-core Model Serving

TGI supports various LLM architectures (see full list [here](./supported_models)). If you wish to serve a model that is not one of the supported models, TGI will fallback to transformers implementation of that model. This means you will be unable to use some of the features introduced by TGI, such as tensor-parallel sharding or flash attention. However, you can still get many benefits of TGI, such as continuous batching or streaming outputs.

You can serve these models using Docker like below 👇 

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2
```

If the model you wish to serve is not a transformers model, but weights and implementation is included in the repository, you can still serve the model by passing `--trust-remote-code` flag to `docker run` command like below 👇 

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <CUSTOM_MODEL_ID> --trust-remote-code
```
