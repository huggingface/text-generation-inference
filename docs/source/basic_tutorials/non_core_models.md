# Non-core Model Serving

TGI supports various LLM architectures (see full list [here](https://github.com/huggingface/text-generation-inference#optimized-architectures)). If you wish to serve a model that is not one of the supported models, TGI will fallback to transformers implementation of that model. They can be loaded by:

```python
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")``

#or

AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")
```

This means you will be unable to use some of the features introduced by TGI, such as tensor-parallel sharding or flash attention. However, you can still get many benefits of TGI, such as continuous batching or streaming outputs.

You can serve these models using Docker like below 👇 

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2
```
