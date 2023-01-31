import concurrent
import os
import re
import torch
import torch.distributed

from datetime import timedelta

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import LocalEntryNotFoundError
from tqdm import tqdm
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizerBase
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)

from text_generation.pb import generate_pb2


class Sampling:
    def __init__(self, seed: int, device: str = "cpu"):
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)
        self.seed = seed

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(
            probs, num_samples=1, generator=self.generator
        ).squeeze(1)
        return next_tokens


class Greedy:
    def __call__(self, logits):
        return logits.argmax(dim=-1)


class NextTokenChooser:
    def __init__(
        self,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=False,
        seed=0,
        device="cpu",
    ):
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
        self.choice = Sampling(seed, device) if sampling else Greedy()

    def __call__(self, input_ids, scores):
        # Warp logits
        scores = self.warpers(input_ids, scores)
        # Compute logprobs
        logprobs = torch.log_softmax(scores, -1)
        # Choose tokens
        next_ids = self.choice(scores)
        return next_ids, logprobs

    @classmethod
    def from_pb(
        cls, pb: generate_pb2.NextTokenChooserParameters, device: torch.device
    ) -> "NextTokenChooser":
        return NextTokenChooser(
            temperature=pb.temperature,
            top_k=pb.top_k,
            top_p=pb.top_p,
            do_sample=pb.do_sample,
            seed=pb.seed,
            device=str(device),
        )


class StopSequenceCriteria:
    def __init__(self, stop_sequence: str):
        self.regex = re.compile(f".*{stop_sequence}$")

    def __call__(self, output: str) -> bool:
        if self.regex.findall(output):
            return True
        return False


class StoppingCriteria:
    def __init__(
        self,
        eos_token_id: int,
        stop_sequence_criterias: List[StopSequenceCriteria],
        max_new_tokens=20,
    ):
        self.eos_token_id = eos_token_id
        self.stop_sequence_criterias = stop_sequence_criterias
        self.max_new_tokens = max_new_tokens
        self.current_tokens = 0
        self.current_output = ""

    def __call__(self, last_token: int, last_output: str) -> Tuple[bool, Optional[str]]:
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True, "length"

        if last_token == self.eos_token_id:
            return True, "eos_token"

        self.current_output += last_output
        for stop_sequence_criteria in self.stop_sequence_criterias:
            if stop_sequence_criteria(self.current_output):
                return True, "stop_sequence"

        return False, None

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.StoppingCriteriaParameters,
        tokenizer: PreTrainedTokenizerBase,
    ) -> "StoppingCriteria":
        stop_sequence_criterias = [
            StopSequenceCriteria(sequence) for sequence in pb.stop_sequences
        ]
        return StoppingCriteria(
            tokenizer.eos_token_id, stop_sequence_criterias, pb.max_new_tokens
        )


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
    )

    return torch.distributed.distributed_c10d._get_default_group(), rank, world_size


def weight_hub_files(model_name, extension=".safetensors"):
    """Get the safetensors filenames on the hub"""
    api = HfApi()
    info = api.model_info(model_name)
    filenames = [s.rfilename for s in info.siblings if s.rfilename.endswith(extension)]
    return filenames


def weight_files(model_name, extension=".safetensors"):
    """Get the local safetensors filenames"""
    filenames = weight_hub_files(model_name, extension)
    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(model_name, filename=filename)
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_name} not found in "
                f"{os.getenv('HUGGINGFACE_HUB_CACHE', 'the local cache')}. "
                f"Please run `text-generation-server download-weights {model_name}` first."
            )
        files.append(cache_file)

    return files


def download_weights(model_name, extension=".safetensors"):
    """Download the safetensors files from the hub"""
    filenames = weight_hub_files(model_name, extension)

    download_function = partial(
        hf_hub_download,
        repo_id=model_name,
        local_files_only=False,
    )

    executor = ThreadPoolExecutor(max_workers=5)
    futures = [
        executor.submit(download_function, filename=filename) for filename in filenames
    ]
    files = [
        future.result()
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))
    ]

    return files
