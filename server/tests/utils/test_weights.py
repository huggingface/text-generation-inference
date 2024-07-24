import pytest
import torch
from text_generation_server.utils.weights import (
    DefaultWeightsLoader,
    UnquantizedWeight,
    Weights,
    WeightsLoader,
)
from text_generation_server.layers.gptq import GPTQWeight, GPTQWeightsLoader
from text_generation_server.layers.exl2 import Exl2Weight, Exl2WeightsLoader
from text_generation_server.layers.marlin.marlin import (
    MarlinWeight,
    MarlinWeightsLoader,
)
from types import SimpleNamespace
from typing import List, Optional, Dict, Union
from pathlib import Path


@pytest.fixture
def gptq_weights_loader():
    return GPTQWeightsLoader(
        bits=4,
        groupsize=-1,
        desc_act=False,
        quant_method="gptq",
        quantize="gptq",
        sym=True,
    )


@pytest.fixture
def gptq_weights_loader_awq():
    return GPTQWeightsLoader(
        bits=4,
        groupsize=-1,
        desc_act=False,
        quant_method="awq",
        quantize="awq",
        sym=True,
    )


@pytest.fixture
def marlin_weights_loader():
    return MarlinWeightsLoader(bits=4, is_marlin_24=False)


dummy_file_system = {
    "test_weights": {
        "layer.0.weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
            dtype=torch.float32,
        ),
    },
    "test_weights_2": {
        "layer.1337.weight": torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=torch.float32,
        ),
    },
    "test_get_weights_col_packed": {
        "weight.weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
    },
    "test_get_multi_weights_col": {
        "weight.weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
        "weight.weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
    },
    "test_get_weights_row": {
        "weight.weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
    },
    "test_get_weights_col_gptq": {
        "weight.qweight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
        "weight.g_idx": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        "weight.qzeros": torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.int32,
        ),
        "weight.scales": torch.tensor(
            [
                [100.0, 100.0],
                [100.0, 100.0],
            ],
            dtype=torch.float16,
        ),
        "gptq_bits": torch.tensor([8], dtype=torch.float32),
        "gptq_groupsize": torch.tensor([2], dtype=torch.float32),
    },
    "test_get_weights_col_marlin": {
        "weight.B": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "weight.s": torch.tensor([[0.5000], [0.2500]], dtype=torch.float16),
    },
    "test_get_weights_row_gptq": {
        "weight.qweight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.int32,
        ),
        "weight.g_idx": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        "weight.qzeros": torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.int32,
        ),
        "weight.scales": torch.tensor(
            [
                [100.0, 100.0],
                [100.0, 100.0],
            ],
            dtype=torch.float16,
        ),
        "gptq_bits": torch.tensor([8], dtype=torch.float32),
        "gptq_groupsize": torch.tensor([2], dtype=torch.float32),
    },
    "test_get_multi_weights_col_gptq": {
        "weight.qweight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.int32,
        ),
        "weight.g_idx": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        "weight.qzeros": torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.int32,
        ),
        "weight.scales": torch.tensor(
            [
                [100.0, 100.0],
                [100.0, 100.0],
            ],
            dtype=torch.float16,
        ),
        "gptq_bits": torch.tensor([8], dtype=torch.float32),
        "gptq_groupsize": torch.tensor([2], dtype=torch.float32),
    },
    "test_get_weights_col_packed_gptq": {
        "weight.qweight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.int32,
        ),
        "weight.g_idx": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        "weight.qzeros": torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.int32,
        ),
        "weight.scales": torch.tensor(
            [
                [100.0, 100.0],
                [100.0, 100.0],
            ],
            dtype=torch.float16,
        ),
        "gptq_bits": torch.tensor([8], dtype=torch.float32),
        "gptq_groupsize": torch.tensor([2], dtype=torch.float32),
    },
    "test_get_weights_col_packed_exl2": {
        "weight.q_weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.int32,
        ),
        "weight.q_scale": torch.tensor([8], dtype=torch.int32),
        "weight.q_invperm": torch.tensor([1, 0, 3, 2], dtype=torch.int32),
        "weight.q_scale_max": torch.tensor([100], dtype=torch.float16),
        "weight.q_groups": torch.tensor([4], dtype=torch.int16),
    },
    "test_get_weights_row_exl2": {
        "weight.q_weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.int32,
        ),
        "weight.q_scale": torch.tensor([8], dtype=torch.int32),
        "weight.q_invperm": torch.tensor([1, 0, 3, 2], dtype=torch.int32),
        "weight.q_scale_max": torch.tensor([100], dtype=torch.float16),
        "weight.q_groups": torch.tensor([4], dtype=torch.int16),
    },
    "test_get_multi_weights_col_exl2": {
        "weight.q_weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.int32,
        ),
        "weight.q_scale": torch.tensor([8], dtype=torch.int32),
        "weight.q_invperm": torch.tensor([1, 0, 3, 2], dtype=torch.int32),
        "weight.q_scale_max": torch.tensor([100], dtype=torch.float16),
        "weight.q_groups": torch.tensor([4], dtype=torch.int16),
    },
    "test_get_weights_col_exl2": {
        "weight.q_weight": torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.int32,
        ),
        "weight.q_scale": torch.tensor([8], dtype=torch.int32),
        "weight.q_invperm": torch.tensor([1, 0, 3, 2], dtype=torch.int32),
        "weight.q_scale_max": torch.tensor([100], dtype=torch.float16),
        "weight.q_groups": torch.tensor([4], dtype=torch.int16),
    },
    "test_get_weights_row_marlin": {
        "weight.B": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "weight.s": torch.tensor([[0.5], [0.25]], dtype=torch.float16),
    },
    "test_get_multi_weights_col_marlin": {
        "weight.B": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "weight.s": torch.tensor([[0.5], [0.25]], dtype=torch.float16),
    },
    "test_get_weights_col_packed_marlin": {
        "weight.B": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "weight.s": torch.tensor([[0.5], [0.25]], dtype=torch.float16),
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
        weights_loader: Optional[WeightsLoader] = None,
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
        self.weights_loader = (
            # We don't need to get linear layers, so just wrap raw tensors.
            DefaultWeightsLoader(lambda x: x)
            if weights_loader is None
            else weights_loader
        )
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


def test_weights():
    weights = MockWeights(
        [
            "test_weights",
            "test_weights_2",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )
    assert weights.get_shape("layer.0.weight") == (2, 2)
    assert weights.get_tensor("layer.1337.weight").shape == (2, 4)


def test_get_tensor():
    weights = MockWeights(
        [
            "test_weights",
            "test_weights_2",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )
    assert torch.allclose(
        weights.get_tensor("layer.0.weight"),
        torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        weights.get_tensor("layer.1337.weight"),
        torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=torch.float32,
        ),
    )


def test_get_weights_col_packed():

    weights = MockWeights(
        [
            "test_get_weights_col_packed",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )

    prefix = "weight"
    block_sizes = 1

    w = weights.get_weights_col_packed(
        prefix=prefix,
        block_sizes=block_sizes,
    )

    assert torch.allclose(
        w,
        torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
    )


def test_get_weights_col_packed_block_size():

    weights = MockWeights(
        [
            "test_get_weights_col_packed",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )

    prefix = "weight"
    block_sizes = 2

    w = weights.get_weights_col_packed(
        prefix=prefix,
        block_sizes=block_sizes,
    )

    assert torch.allclose(
        w,
        torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
    )


def test_get_weights_col_packed_block_size_arr():

    weights = MockWeights(
        [
            "test_get_weights_col_packed",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )

    prefix = "weight"
    block_sizes = [1, 1]

    w = weights.get_weights_col_packed(
        prefix=prefix,
        block_sizes=block_sizes,
    )

    assert torch.allclose(
        w,
        torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
    )


def test_get_multi_weights_col():
    weights = MockWeights(
        [
            "test_get_multi_weights_col",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )

    prefixes = ["weight", "weight"]

    w = weights.get_multi_weights_col(
        prefixes=prefixes,
        dim=0,
    )

    assert torch.allclose(
        w,
        torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=torch.float32,
        ),
    )


def test_get_weights_row():
    weights = MockWeights(
        [
            "test_get_weights_row",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
    )

    prefix = "weight"

    w = weights.get_weights_row(
        prefix=prefix,
    )

    assert torch.allclose(
        w,
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
        ),
    )


# test_get_weights_col


def test_get_weights_col_awq(gptq_weights_loader_awq):
    weights = MockWeights(
        [
            "test_get_weights_col_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader_awq,
    )

    prefix = "weight"

    w = weights.get_weights_col(
        prefix=prefix,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor(
            [[100.0, 100.0], [100.0, 100.0]],
            dtype=torch.float16,
        ),
        g_idx=None,
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=True,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert w.g_idx == expected_weight.g_idx, "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


def test_get_weights_col_gtpq(gptq_weights_loader):
    weights = MockWeights(
        [
            "test_get_weights_col_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader,
    )

    prefix = "weight"

    w = weights.get_weights_col(
        prefix=prefix,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor([[100.0, 100.0], [100.0, 100.0]], dtype=torch.float16),
        g_idx=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=False,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert torch.allclose(w.g_idx, expected_weight.g_idx), "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


def test_get_weights_col_exl2():
    weights = MockWeights(
        [
            "test_get_weights_col_exl2",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=Exl2WeightsLoader(),
    )

    prefix = "weight"

    w = weights.get_weights_col(
        prefix=prefix,
    )

    scaled_scale_max = 0.3906 * 256
    expected_weight = Exl2Weight(
        q_weight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        q_scale=torch.tensor([8], dtype=torch.int32),
        q_invperm=torch.tensor([1, 0, 3, 2], dtype=torch.int16),
        q_scale_max=torch.tensor([scaled_scale_max], dtype=torch.float16),
        q_groups=torch.tensor([4], dtype=torch.int16),
    )

    assert torch.allclose(w.q_weight, expected_weight.q_weight), "q_weight mismatch"
    assert torch.allclose(w.q_scale, expected_weight.q_scale), "q_scale mismatch"
    assert torch.allclose(w.q_invperm, expected_weight.q_invperm), "q_invperm mismatch"
    assert torch.allclose(
        w.q_scale_max, expected_weight.q_scale_max
    ), "q_scale_max mismatch"
    assert torch.allclose(w.q_groups, expected_weight.q_groups), "q_groups mismatch"


def test_get_weights_col_marlin(marlin_weights_loader):
    weights = MockWeights(
        [
            "test_get_weights_col_marlin",
        ],
        device="cpu",
        dtype=torch.float16,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=marlin_weights_loader,
    )

    prefix = "weight"

    w = weights.get_weights_col(
        prefix=prefix,
    )

    expected_weight = MarlinWeight(
        B=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        s=torch.tensor([[0.5000], [0.2500]], dtype=torch.float16),
    )

    assert torch.allclose(w.B, expected_weight.B), "B mismatch"
    assert torch.allclose(w.s, expected_weight.s), "s mismatch"


# test_get_weights_col_packed


def test_get_weights_col_packed_awq(gptq_weights_loader_awq):
    weights = MockWeights(
        [
            "test_get_weights_col_packed_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader_awq,
    )

    prefix = "weight"
    block_sizes = 1

    w = weights.get_weights_col_packed(
        prefix=prefix,
        block_sizes=block_sizes,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor([[100.0, 100.0], [100.0, 100.0]], dtype=torch.float16),
        g_idx=None,
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=True,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert w.g_idx == expected_weight.g_idx, "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


@pytest.mark.skip(reason="Review expected functionality")
def test_get_weights_col_packed_exl2():
    weights = MockWeights(
        [
            "test_get_weights_col_packed_exl2",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=Exl2WeightsLoader(),
    )

    prefix = "weight"
    block_sizes = 1

    w = weights.get_weights_col_packed(
        prefix=prefix,
        block_sizes=block_sizes,
    )

    scaled_scale_max = 0.3906 * 256
    expected_weight = Exl2Weight(
        q_weight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        q_scale=torch.tensor([8], dtype=torch.int32),
        q_invperm=torch.tensor([1], dtype=torch.int16),
        q_scale_max=torch.tensor([scaled_scale_max], dtype=torch.float16),
        q_groups=torch.tensor([4], dtype=torch.int16),
    )

    assert torch.allclose(w.q_weight, expected_weight.q_weight), "q_weight mismatch"
    assert torch.allclose(w.q_scale, expected_weight.q_scale), "q_scale mismatch"
    assert torch.allclose(w.q_invperm, expected_weight.q_invperm), "q_invperm mismatch"
    assert torch.allclose(
        w.q_scale_max, expected_weight.q_scale_max
    ), "q_scale_max mismatch"
    assert torch.allclose(w.q_groups, expected_weight.q_groups), "q_groups mismatch"


def test_get_weights_col_packed_gptq(gptq_weights_loader):
    weights = MockWeights(
        [
            "test_get_weights_col_packed_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader,
    )

    prefixes = ["weight"]

    w = weights.get_multi_weights_col(
        prefixes=prefixes,
        dim=0,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor([[100.0, 100.0], [100.0, 100.0]], dtype=torch.float16),
        g_idx=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=False,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert torch.allclose(w.g_idx, expected_weight.g_idx), "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


def test_get_weights_col_packed_marlin(marlin_weights_loader):
    weights = MockWeights(
        [
            "test_get_weights_col_packed_marlin",
        ],
        device="cpu",
        dtype=torch.float16,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=marlin_weights_loader,
    )

    prefix = "weight"

    w = weights.get_multi_weights_col(
        prefixes=[prefix],
        dim=0,
    )

    expected_weight = MarlinWeight(
        B=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        s=torch.tensor([[0.5000], [0.2500]], dtype=torch.float16),
    )

    print(expected_weight)

    assert torch.allclose(w.B, expected_weight.B), "B mismatch"
    assert torch.allclose(w.s, expected_weight.s), "s mismatch"


# test_get_multi_weights_col


def test_get_multi_weights_col_awq(gptq_weights_loader_awq):
    weights = MockWeights(
        [
            "test_get_multi_weights_col_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader_awq,
    )

    prefixes = ["weight"]

    w = weights.get_multi_weights_col(
        prefixes=prefixes,
        dim=0,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor([[100.0, 100.0], [100.0, 100.0]], dtype=torch.float16),
        g_idx=None,
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=True,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert w.g_idx == expected_weight.g_idx, "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


def test_get_multi_weights_col_exl2():
    weights = MockWeights(
        [
            "test_get_multi_weights_col_exl2",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=Exl2WeightsLoader(),
    )

    prefix = "weight"

    try:
        w = weights.get_multi_weights_col(
            prefixes=[prefix],
            dim=0,
        )
    except ValueError as e:
        assert e.args[0] == "get_multi_weights_col is not supported for exl2"


def test_get_multi_weights_col_gptq(gptq_weights_loader):
    weights = MockWeights(
        [
            "test_get_multi_weights_col_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader,
    )

    prefixes = ["weight"]

    w = weights.get_multi_weights_col(
        prefixes=prefixes,
        dim=0,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor([[100.0, 100.0], [100.0, 100.0]], dtype=torch.float16),
        g_idx=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=False,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert torch.allclose(w.g_idx, expected_weight.g_idx), "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


def test_get_multi_weights_col_marlin(marlin_weights_loader):
    weights = MockWeights(
        [
            "test_get_multi_weights_col_marlin",
        ],
        device="cpu",
        dtype=torch.float16,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=marlin_weights_loader,
    )

    prefix = "weight"

    w = weights.get_multi_weights_col(
        prefixes=[prefix],
        dim=0,
    )

    expected_weight = MarlinWeight(
        B=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        s=torch.tensor([[0.5000], [0.2500]], dtype=torch.float16),
    )

    assert torch.allclose(w.B, expected_weight.B), "B mismatch"
    assert torch.allclose(w.s, expected_weight.s), "s mismatch"


# test_get_weights_row


def test_get_weights_row_awq(gptq_weights_loader_awq):
    weights = MockWeights(
        [
            "test_get_weights_row_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader_awq,
    )

    prefix = "weight"

    w = weights.get_weights_row(
        prefix=prefix,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor([[100.0, 100.0], [100.0, 100.0]], dtype=torch.float16),
        g_idx=None,
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=True,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert w.g_idx == expected_weight.g_idx, "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


def test_get_weights_row_exl2():
    weights = MockWeights(
        [
            "test_get_weights_row_exl2",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=Exl2WeightsLoader(),
    )

    prefix = "weight"

    w = weights.get_weights_row(
        prefix=prefix,
    )
    print(w)

    scaled_scale_max = 0.3906 * 256
    expected_weight = Exl2Weight(
        q_weight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        q_scale=torch.tensor([8], dtype=torch.int32),
        q_invperm=torch.tensor([1, 0, 3, 2], dtype=torch.int16),
        q_scale_max=torch.tensor([scaled_scale_max], dtype=torch.float16),
        q_groups=torch.tensor([4], dtype=torch.int16),
    )

    assert torch.allclose(w.q_weight, expected_weight.q_weight), "q_weight mismatch"
    assert torch.allclose(w.q_scale, expected_weight.q_scale), "q_scale mismatch"
    assert torch.allclose(w.q_invperm, expected_weight.q_invperm), "q_invperm mismatch"
    assert torch.allclose(
        w.q_scale_max, expected_weight.q_scale_max
    ), "q_scale_max mismatch"
    assert torch.allclose(w.q_groups, expected_weight.q_groups), "q_groups mismatch"


def test_get_weights_row_gptq(gptq_weights_loader):
    weights = MockWeights(
        [
            "test_get_weights_row_gptq",
        ],
        device="cpu",
        dtype=torch.float32,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=gptq_weights_loader,
    )

    prefix = "weight"

    w = weights.get_weights_row(
        prefix=prefix,
    )

    expected_weight = GPTQWeight(
        qweight=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        qzeros=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        scales=torch.tensor([[100.0, 100.0], [100.0, 100.0]], dtype=torch.float16),
        g_idx=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        bits=8.0,
        groupsize=2.0,
        use_awq_kernel=False,
        use_exllama=False,
    )

    assert torch.allclose(w.qweight, expected_weight.qweight), "qweight mismatch"
    assert torch.allclose(w.qzeros, expected_weight.qzeros), "qzeros mismatch"
    assert torch.allclose(w.scales, expected_weight.scales), "scales mismatch"
    assert torch.allclose(w.g_idx, expected_weight.g_idx), "g_idx mismatch"
    assert w.bits == expected_weight.bits, "bits mismatch"
    assert w.groupsize == expected_weight.groupsize, "groupsize mismatch"
    assert w.use_awq_kernel == expected_weight.use_awq_kernel, "use_awq_kernel mismatch"
    assert w.use_exllama == expected_weight.use_exllama, "use_exllama mismatch"


def test_get_weights_row_marlin(marlin_weights_loader):
    weights = MockWeights(
        [
            "test_get_weights_row_marlin",
        ],
        device="cpu",
        dtype=torch.float16,
        process_group=dummy_process_group,
        dummy_fs=dummy_file_system,
        weights_loader=marlin_weights_loader,
    )

    prefix = "weight"

    w = weights.get_weights_row(
        prefix=prefix,
    )

    expected_weight = MarlinWeight(
        B=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        s=torch.tensor([[0.5000], [0.2500]], dtype=torch.float16),
    )

    assert torch.allclose(w.B, expected_weight.B), "B mismatch"
    assert torch.allclose(w.s, expected_weight.s), "s mismatch"
