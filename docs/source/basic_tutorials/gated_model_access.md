### Serving Private & Gated Models

If the model you wish to serve is behind gated access or the model repository on Hugging Face Hub is private, and you have the access to the model, you can provide your Hugging Face Hub access token. To do so, simply head to [Hugging Face Hub tokens page](https://huggingface.co/settings/tokens), copy a token with READ access and export `HUGGING_FACE_HUB_TOKEN=<YOUR READ TOKEN>` in CLI.

If you would like to do it through Docker, you can provide your token like below.

```shell
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your cli READ token>

docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.0 --model-id $model
```