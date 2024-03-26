# TGI-Gaudi example

This example provide a simple way of usage of `tgi-gaudi` with continuous batching. It uses a small dataset [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) and present basic performance numbers.

## Get started

### Install

```
pip install -r requirements
```

### Setup TGI server

More details on runing the TGI server available [here](https://github.com/huggingface/tgi-gaudi/blob/habana-main/README.md#running-tgi-on-gaudi).

### Run benchmark

To run benchmark use below command:

```
python run_generation --model_id MODEL_ID
```
where `MODEL_ID` should be set to the same value as in the TGI server instance.
> For gated models such as [LLama](https://huggingface.co/meta-llama) or [StarCoder](https://huggingface.co/bigcode/starcoder), you will have to set environment variable `HUGGING_FACE_HUB_TOKEN=<token>` with a valid Hugging Face Hub read token.

All possible parameters are described in the below table:
<div align="left">

| Name                      | Default value                 | Description                                                   |
| ------------------------- | :---------------------------- | :------------------------------------------------------------ |
| SERVER_ADDRESS            | http://localhost:8080         | The address and port at which the TGI server is available.    |
| MODEL_ID                  | meta-llama/Llama-2-7b-chat-hf | Model ID used in the TGI server instance.                     |
| MAX_INPUT_LENGTH          | 1024                          | Maximum input length supported by the TGI server.             |
| MAX_OUTPUT_LENGTH         | 1024                          | Maximum output length supported by the TGI server.            |
| TOTAL_SAMPLE_COUNT        | 2048                          | Number of samples to run.                                     |
| MAX_CONCURRENT_REQUESTS   | 256                           | The number of requests sent simultaneously to the TGI server. |

</div>