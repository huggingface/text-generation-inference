# Using TGI through CLI

You can use CLI tools of TGI to download weights, serve and quantize models, or get information on serving parameters. 

## Installing TGI for CLI

To install TGI to use with CLI, you need to first clone the TGI repository, then inside the repository, run

```shell
make install
```

If you would like to serve models with custom kernels, run

```shell
BUILD_EXTENSIONS=True make install
```

After running this, you will be able to use `text-generation-server` and `text-generation-launcher`.

`text-generation-server` lets you download the model with `download-weights` command like below 👇 

```shell
text-generation-server download-weights MODEL_HUB_ID
```

You can also use it to quantize models like below 👇 

```shell
text-generation-server quantize MODEL_HUB_ID OUTPUT_DIR 
```

You can use `text-generation-launcher` to serve models. 

```shell
text-generation-launcher --model-id MODEL_HUB_ID --port 8080
```

There are many options and parameters you can pass to `text-generation-launcher`. The documentation for CLI is kept minimal and intended to rely on self-generating documentation, which can be found by running 

```shell
text-generation-launcher --help
``` 

You can also find it hosted in this [Swagger UI](https://huggingface.github.io/text-generation-inference/).

Same documentation can be found for `text-generation-server`.

```shell
text-generation-server --help
``````


