# Serving Private & Gated Models

If the model you wish to serve is behind gated access or the model repository on Hugging Face Hub is private, and you have access to the model, you can provide your Hugging Face Hub access token. You can generate and copy a read token from [Hugging Face Hub tokens page](https://huggingface.co/settings/tokens)

If you're using the CLI, set the `HF_TOKEN` environment variable. For example:

```
export HF_TOKEN=<YOUR READ TOKEN>
```

If you would like to do it through Docker, you can provide your token by specifying `HF_TOKEN` as shown below.

```bash
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data
token=<your READ token>

docker run --gpus all \
    --shm-size 1g \
    -e HF_TOKEN=$token \
    -p 8080:80 \
    -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id $model
```
