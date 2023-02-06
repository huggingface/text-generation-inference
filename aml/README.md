# Azure ML endpoint

## Create all resources

```shell
az ml model create -f model.yaml -g HuggingFace-BLOOM-ModelPage -w HuggingFace
az ml online-endpoint create -f endpoint.yaml -g HuggingFace-BLOOM-ModelPage -w HuggingFace
az ml online-deployment create -f deployment.yaml -g HuggingFace-BLOOM-ModelPage -w HuggingFace
```

## Update deployment

```shell
az ml online-deployment update -f deployment.yaml -g HuggingFace-BLOOM-ModelPage -w HuggingFace
```