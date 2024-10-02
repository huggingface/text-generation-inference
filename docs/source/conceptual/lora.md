# LoRA (Low-Rank Adaptation)

## What is LoRA?

LoRA is a technique that allows for efficent fine-tuning a model while only updating a small portion of the model's weights. This is useful when you have a large model that has been pre-trained on a large dataset, but you want to fine-tune it on a smaller dataset or for a specific task.

LoRA works by adding a small number of additional weights to the model, which are used to adapt the model to the new dataset or task. These additional weights are learned during the fine-tuning process, while the rest of the model's weights are kept fixed.

## How is it used?

LoRA can be used in many ways and the community is always finding new ways to use it. Here are some examples of how you can use LoRA:

Technically, LoRA can be used to fine-tune a large language model on a small dataset. However, these use cases can span a wide range of applications, such as:

- fine-tuning a language model on a small dataset
- fine-tuning a language model on a domain-specific dataset
- fine-tuning a language model on a dataset with limited labels

## Optimizing Inference with LoRA

LoRA's can be used during inference by mutliplying the adapter weights with the model weights at each specified layer. This process can be computationally expensive, but due to awesome work by [punica-ai](https://github.com/punica-ai/punica) and the [lorax](https://github.com/predibase/lorax) team, optimized kernels/and frameworks have been developed to make this process more efficient. TGI leverages these optimizations in order to provide fast and efficient inference with mulitple LoRA models.

## Serving multiple LoRA adapters with TGI

Once a LoRA model has been trained, it can be used to generate text or perform other tasks just like a regular language model. However, because the model has been fine-tuned on a specific dataset, it may perform better on that dataset than a model that has not been fine-tuned.

In practice its often useful to have multiple LoRA models, each fine-tuned on a different dataset or for a different task. This allows you to use the model that is best suited for a particular task or dataset.

Text Generation Inference (TGI) now supports loading multiple LoRA models at startup that can be used in generation requests. This feature is available starting from version `~2.0.6` and is compatible with LoRA models trained using the `peft` library.

### Specifying LoRA models

To use LoRA in TGI, when starting the server, you can specify the list of LoRA models to load using the `LORA_ADAPTERS` environment variable. For example:

```bash
LORA_ADAPTERS=predibase/customer_support,predibase/dbpedia
```

To specify model revision, use `adapter_id@revision`, as follows:

```bash
LORA_ADAPTERS=predibase/customer_support@main,predibase/dbpedia@rev2
```

To use a locally stored lora adapter, use `adapter-name=/path/to/adapter`, as seen below. When you want to use this adapter, set `"parameters": {"adapter_id": "adapter-name"}"`

```bash
LORA_ADAPTERS=myadapter=/some/path/to/adapter,myadapter2=/another/path/to/adapter
```

note it's possible to mix adapter_ids with adapter_id=adapter_path e.g.

```bash
LORA_ADAPTERS=predibase/dbpedia,myadapter=/path/to/dir/
```

In the server logs, you will see the following message:

```txt
Loading adapter weights into model: predibase/customer_support
Loading adapter weights into model: predibase/dbpedia
```

## Generate text

You can then use these models in generation requests by specifying the `lora_model` parameter in the request payload. For example:

```json
curl 127.0.0.1:3000/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
  "inputs": "Hello who are you?",
  "parameters": {
    "max_new_tokens": 40,
    "adapter_id": "predibase/customer_support"
  }
}'
```

If you are using a lora adapter stored locally that was set in the following manner: `LORA_ADAPTERS=myadapter=/some/path/to/adapter`, here is an example payload:

```json
curl 127.0.0.1:3000/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
  "inputs": "Hello who are you?",
  "parameters": {
    "max_new_tokens": 40,
    "adapter_id": "myadapter"
  }
}'
```


> **Note:** The Lora feature is new and still being improved. If you encounter any issues or have any feedback, please let us know by opening an issue on the [GitHub repository](https://github.com/huggingface/text-generation-inference/issues/new/choose). Additionally documentation and an improved client library will be published soon.

An updated tutorial with detailed examples will be published soon. Stay tuned!
