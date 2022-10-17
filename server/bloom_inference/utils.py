import os
import contextlib
import torch
import torch.distributed

from datetime import timedelta
from transformers.generation_logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)


class Sampling:
    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens


class Greedy:
    def __call__(self, logits):
        return logits.argmax(dim=-1)


class NextTokenChooser:
    def __init__(self, temperature=1.0, top_k=None, top_p=None, do_sample=False):
        warpers = LogitsProcessorList()
        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        sampling = do_sample
        if temperature is not None and temperature != 1.0:
            temperature = float(temperature)
            warpers.append(TemperatureLogitsWarper(temperature))
            sampling = True
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
            sampling = True
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
            sampling = True

        self.warpers = warpers
        self.choice = Sampling() if sampling else Greedy()

    def __call__(self, input_ids, scores):
        scores = self.warpers(input_ids, scores)
        next_ids = self.choice(scores)
        return next_ids.unsqueeze(-1)


class StoppingCriteria:
    def __init__(self, max_new_tokens=20):
        self.max_new_tokens = max_new_tokens
        self.current_tokens = 0

    def __call__(self, all_ids):
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True
        return False


def initialize_torch_distributed():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        # initialized `torch.distributed`
        # Set the device id.
        assert world_size <= torch.cuda.device_count(), "Each process is one gpu"
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        backend = "gloo"

    # Call the init process.
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=60),
        init_method="tcp://localhost:6000",
    )

    return torch.distributed.distributed_c10d._get_default_group(), rank, world_size


@contextlib.contextmanager
def set_default_dtype(dtype):
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved_dtype)
