```shell
docker build . -t db4c2190dd824d1f950f5d1555fbadf0.azurecr.io/text-generation:0.1
docker push db4c2190dd824d1f950f5d1555fbadf0.azurecr.io/text-generation:0.1

az ml online-endpoint create -f endpoint.yaml -g HuggingFace-BLOOM-ModelPage -w HuggingFace
az ml online-deployment create -f deployment.yaml -g HuggingFace-BLOOM-ModelPage -w HuggingFace
```