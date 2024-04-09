# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import os
import numpy as np
import pytest
import torch
from text_generation_server.utils.watermark import WatermarkLogitsProcessor


GAMMA = os.getenv("WATERMARK_GAMMA", 0.5)
DELTA = os.getenv("WATERMARK_DELTA", 2.0)


@pytest.fixture
def hpu_device():
    return torch.device("hpu")


@pytest.fixture
def input_ids_list():
    return [101, 2036, 3731, 102, 2003, 103]


@pytest.fixture
def input_ids_tensor(hpu_device):
    return torch.tensor(
        [[101, 2036, 3731, 102, 2003, 103]],
        dtype=torch.int64,
        device=hpu_device
    )


@pytest.fixture
def scores(hpu_device):
    return torch.tensor(
        [[0.5, 0.3, 0.2, 0.8], [0.1, 0.2, 0.7, 0.9]],
        device=hpu_device
    )


def test_seed_rng(input_ids_list, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    processor._seed_rng(input_ids_list)
    assert isinstance(processor.rng, torch.Generator)


def test_seed_rng_tensor(input_ids_tensor, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    processor._seed_rng(input_ids_tensor)
    assert isinstance(processor.rng, torch.Generator)


def test_get_greenlist_ids(input_ids_list, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    result = processor._get_greenlist_ids(input_ids_list, 10, hpu_device)
    assert max(result) <= 10
    assert len(result) == int(10 * 0.5)


def test_get_greenlist_ids_tensor(input_ids_tensor, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    result = processor._get_greenlist_ids(input_ids_tensor, 10, hpu_device)
    assert max(result) <= 10
    assert len(result) == int(10 * 0.5)


def test_calc_greenlist_mask(scores, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    greenlist_token_ids = torch.tensor([2, 3], device=hpu_device)
    result = processor._calc_greenlist_mask(scores, greenlist_token_ids)
    assert result.tolist() == [[False, False, False, False], [False, False, True, True]]
    assert result.shape == scores.shape


def test_bias_greenlist_logits(scores, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    green_tokens_mask = torch.tensor(
        [[False, False, True, True], [False, False, False, True]], device=hpu_device
    )
    greenlist_bias = 2.0
    result = processor._bias_greenlist_logits(scores, green_tokens_mask, greenlist_bias)
    assert np.allclose(result.tolist(), [[0.5, 0.3, 2.2, 2.8], [0.1, 0.2, 0.7, 2.9]])
    assert result.shape == scores.shape


def test_call(input_ids_list, scores, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    result = processor(input_ids_list, scores)
    assert result.shape == scores.shape


def test_call_tensor(input_ids_tensor, scores, hpu_device):
    processor = WatermarkLogitsProcessor(device=hpu_device)
    result = processor(input_ids_tensor, scores)
    assert result.shape == scores.shape
