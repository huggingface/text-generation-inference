import asyncio
import logging
import time
from text_generation import AsyncClient

from typing import List, Dict
from torch import IntTensor
from transformers import AutoTokenizer
import numpy as np

from hydra import (
    compose,
    initialize
)
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger(__name__)
# intialize Hydra subsystem
initialize(version_base=None, config_path="conf")
cfg: DictConfig = compose("config.yaml")
print(f"Experiment configuration:\n{OmegaConf.to_yaml(cfg)}")

# serving system deployment
max_concurrent_requests: int = cfg.deployment.max_concurrent_requests
endpoint: str = ""
if cfg.deployment.local: endpoint = f"{cfg.deployment.addr}:{ cfg.deployment.port}"
if not cfg.deployment.local: endpoint = cfg.deployment.endpoint
client = AsyncClient(endpoint)

# uncomment for using natural language prompts
# prompt: str = "Hello" * 100
# output_lenght (decoding length)
max_new_tokens: int = cfg.GenerationConfig.max_new_tokens
# generation_strategy
repetition_penalty: float = cfg.GenerationConfig.repetition_penalty
do_sample: bool = cfg.GenerationConfig.do_sample
# generation finish reason
stop_sequences: List[str] = [stop for stop in cfg.GenerationConfig.stop_sequences]
tokenizer = AutoTokenizer.from_pretrained(cfg.macros.tokenizer_name)
# "id": 21820 = "Hello"
dummy_input: IntTensor = IntTensor([[21820]])
# inverse tokenization (sanity check)
# token: str = tokenizer.decode(dummy_input[0], skip_special_tokens=True)
sequence_length: int = cfg.GenerationConfig.sequence_length
multiple_dummy_repeat = dummy_input.repeat(1, sequence_length)
prompt: str = tokenizer.decode(multiple_dummy_repeat[0], skip_special_tokens=True)
assert np.shape(multiple_dummy_repeat)[-1] == sequence_length

generate_kwargs: Dict = {"do_sample": do_sample,
                         "max_new_tokens": max_new_tokens,
                        "repetition_penalty": repetition_penalty,
                        "stop_sequences": stop_sequences,
                        "decoder_input_details": True,
                        }

prompts: List = [prompt] * max_concurrent_requests
# create many coroutines
coros = [client.generate(prompt, **generate_kwargs) for prompt in prompts]

async def batch():
    return await asyncio.gather(*coros)
    
st: float = time.perf_counter_ns()
results = asyncio.run(batch())
et = (time.perf_counter_ns() - st) * 1e-9
print(f"Serving elapsed time: {et:0.4f} seconds")
# check the last response
print(results[-1].details)

total_input_sequence_tokens: int = 0
total_decoded_tokens: int = 0

for prompt, response in zip(prompts,results):
    # uncomment for see generations
    # print(prompt + response.generated_text)
    # assert np.shape(tokenizer(response.generated_text, return_tensors="pt").input_ids)[-1] -1 == max_new_tokens
    # assert response.details.generated_tokens == max_new_tokens
    total_input_sequence_tokens += len(response.details.prefill)
    total_decoded_tokens += response.details.generated_tokens
assert total_input_sequence_tokens == max_concurrent_requests * sequence_length

# stats
print(f"Serving elapsed time: {et:0.4f} seconds")
print(f"Total requests: {max_concurrent_requests}")
print(f"Total sequences tokens: {total_input_sequence_tokens}")
print(f"Sequence length: {sequence_length}")
print(f"Total decoded tokens: {total_decoded_tokens}")
print(f"Throughput: {total_decoded_tokens/et} tokens/sec")
print(f"Total processed tokens: {total_decoded_tokens + total_input_sequence_tokens}")