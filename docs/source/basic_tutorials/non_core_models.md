# Non-core Model Serving

TGI supports various LLM architectures (see full list [here](../supported_models)). If you wish to serve a model that is not one of the supported models, TGI will fallback to the `transformers` implementation of that model. This means you will be unable to use some of the features introduced by TGI, such as tensor-parallel sharding or flash attention. However, you can still get many benefits of TGI, such as continuous batching or streaming outputs.

You can serve these models using the same Docker command-line invocation as with fully supported models 👇

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2
```

If the model you wish to serve is a custom transformers model, and its weights and implementation are available in the Hub, you can still serve the model by passing the `--trust-remote-code` flag to the `docker run` command like below 👇

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <CUSTOM_MODEL_ID> --trust-remote-code
```

Finally, if the model is not on Hugging Face Hub but on your local, you can pass the path to the folder that contains your model like below 👇

```bash
# Make sure your model is in the $volume directory
docker run --shm-size 1g -p 8080:80 -v $volume:/data  ghcr.io/huggingface/text-generation-inference:latest --model-id /data/<PATH-TO-FOLDER>
```

You can refer to [transformers docs on custom models](https://huggingface.co/docs/transformers/main/en/custom_models) for more information.
