import torch
from types import SimpleNamespace
from typing import List, Union, Dict, Optional
import time
from pathlib import Path
from text_generation_server.utils.weights import Weights
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaAttention,
)


dummy_file_system = {
    "file1": {
        "prefix.self_attn.q_proj.weight": torch.rand(
            4096, 4096
        ),  # (hidden_size * num_heads, hidden_size)
        "prefix.self_attn.k_proj.weight": torch.rand(4096, 4096),
        "prefix.self_attn.v_proj.weight": torch.rand(4096, 4096),
        "prefix.self_attn.o_proj.weight": torch.rand(
            4096, 4096
        ),  # (hidden_size, hidden_size * num_heads)
        "prefix.self_attn.q_proj.bias": torch.rand(4096),
        "prefix.self_attn.k_proj.bias": torch.rand(4096),
        "prefix.self_attn.v_proj.bias": torch.rand(4096),
        "prefix.self_attn.o_proj.bias": torch.rand(4096),
    },
}


class MockSlice:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_shape(self):
        return self.tensor.shape

    def __getitem__(self, idx):
        return self.tensor[idx]


def mock_get_slice(tensor_name, filename):
    tensor = dummy_file_system[filename][tensor_name]
    return MockSlice(tensor)


def mock_handle(filename, device, dtype):
    return SimpleNamespace(
        get_slice=lambda tensor_name: mock_get_slice(tensor_name, filename)
    )


class MockSafeOpen:
    def __init__(self, filename, framework, dummy_fs):
        self.filename = filename
        self.framework = framework
        self.dummy_fs = dummy_fs

    def keys(self):
        return list(self.dummy_fs[self.filename].keys())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockWeights(Weights):
    def __init__(
        self,
        filenames: List[Union[Path, str]],
        device,
        dtype,
        process_group,
        dummy_fs,
        aliases: Optional[Dict[str, List[str]]] = None,
        prefix: Optional[str] = None,
    ):
        routing = {}
        self.dummy_fs = dummy_fs
        for filename in filenames:
            with MockSafeOpen(filename, framework="pytorch", dummy_fs=dummy_fs) as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    routing[k] = filename
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group
        self.prefix = prefix
        self._handles = {}

    def _get_handle(self, filename: Union[Path, str]):
        if filename in self._handles:
            return self._handles[filename]
        else:
            handle = mock_handle(filename, self.device, self.dtype)
            self._handles[filename] = handle
            return handle

    def get_shape(self, tensor_name: str):
        filename, _ = self.get_filename(tensor_name)
        handle = self._get_handle(filename)
        return handle.get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str):
        filename, _ = self.get_filename(tensor_name)
        handle = self._get_handle(filename)
        return handle.get_slice(tensor_name).tensor


dummy_process_group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)


def run_test(attention, input_tensor, num_warmup=0, num_iterations=1):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # warm-up
    for _ in range(num_warmup):
        attention.query_key_value(input_tensor, None)
        torch.cuda.synchronize()

    # timed
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        attention.query_key_value(input_tensor, None)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # milliseconds

    return sum(times) / len(times)


def test_qkv_batch_speedup(capsys):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hidden_size = 4096
    num_heads = 32

    config = SimpleNamespace(
        num_attention_heads=num_heads,
        hidden_size=hidden_size,
        rope_theta=10000.0,
        num_key_value_heads=num_heads,
        model_type="llama",
        quantize=None,
    )

    weights = MockWeights(
        filenames=["file1"],
        device=device,
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )

    attention = FlashLlamaAttention(
        index=0,
        prefix="prefix.self_attn",
        config=config,
        weights=weights,
    )

    # sequence with various odd lengths
    sequence_lengths = [3, 35, 365, 3_501, 11_111]
    batch_sizes = [*range(16)]

    with capsys.disabled():
        # allow printing
        for sequence_length in sequence_lengths:
            print(f"Testing with sequence length: {sequence_length}")
            print(
                f"{'Batch Size':<10} {'Batch Time (ms)':<20} {'Sequential Time (ms)':<25} {'Speedup':<10}"
            )
            print("-" * 65)

            for batch_size in batch_sizes:
                # batch
                batch_input = torch.rand(
                    sequence_length * batch_size, hidden_size, device=device
                )
                batch_time = run_test(attention, batch_input)

                # sequential
                sequential_time = 0
                for index in range(batch_size):
                    single_input = batch_input[
                        index * sequence_length : (index + 1) * sequence_length
                    ]
                    sequential_time += run_test(attention, single_input)

                speedup = sequential_time / batch_time
                print(
                    f"{batch_size:<10} {batch_time:<20.2f} {sequential_time:<25.2f} {speedup:<10.2f}"
                )
