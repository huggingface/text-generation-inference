import pytest
import torch
from unittest.mock import Mock, patch
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
    DynamicPositionRotaryEmbedding,
    YarnPositionRotaryEmbedding,
)
from text_generation_server.utils.import_utils import SYSTEM


def test_position_rotary_embedding_static_basic():
    config = Mock(
        rope_theta=10000,
        max_position_embeddings=2048,
        rope_scaling=None
    )
    weights = Mock(device=torch.device("cpu"))

    result = PositionRotaryEmbedding.static(
        config=config,
        dim=64,
        base=config.rope_theta,
        device=weights.device,
    )

    assert isinstance(result, PositionRotaryEmbedding)
    assert result.inv_freq.shape == (32,)  # dim // 2
    assert result.scaling_factor is None


def test_position_rotary_embedding_static_linear_scaling():
    config = Mock(
        rope_theta=10000,
        max_position_embeddings=2048
    )
    # scaling is not applied if type is linear (TODO: maybe revisit this)
    config.rope_scaling = {"type": "linear", "factor": 2.0}
    weights = Mock(device=torch.device("cpu"))

    result = PositionRotaryEmbedding.static(
        config=config,
        dim=64,
        base=config.rope_theta,
        device=weights.device,
    )

    assert isinstance(result, PositionRotaryEmbedding)
    assert result.scaling_factor is None


def test_position_rotary_embedding_static_dynamic_scaling():
    config = Mock(
        rope_theta=10000,
        max_position_embeddings=2048,
        rope_scaling = {"type": "dynamic", "factor": 2.0}
    )
    weights = Mock(device=torch.device("cpu"))

    result = PositionRotaryEmbedding.static(
        config=config,
        dim=64,
        base=config.rope_theta,
        device=weights.device,
    )

    assert isinstance(result, DynamicPositionRotaryEmbedding)
    assert result.scaling_factor == 2.0
    assert result.max_position_embeddings == 2048


def test_position_rotary_embedding_static_yarn_scaling():
    config = Mock(
        rope_theta=10000,
        max_position_embeddings=2048,
        rope_scaling = {
            "type": "yarn",
            "factor": 1.5,
            "original_max_position_embeddings": 2048,
        }
    )
    weights = Mock(device=torch.device("cpu"))

    result = PositionRotaryEmbedding.static(
        config=config,
        dim=64,
        base=config.rope_theta,
        device=weights.device,
    )

    assert isinstance(result, YarnPositionRotaryEmbedding)
    assert result.scaling_factor == 1.5
    assert result.max_position_embeddings == 2048


def test_position_rotary_embedding_static_invalid_scaling():
    config = Mock(
        rope_theta=10000,
        max_position_embeddings=2048,
        rope_scaling = {"type": "invalid", "factor": 2.0}
    )
    weights = Mock(device=torch.device("cpu"))

    with pytest.raises(NotImplementedError):
        PositionRotaryEmbedding.static(
            config=config,
            dim=64,
            base=config.rope_theta,
            device=weights.device,
        )


def test_position_rotary_embedding_static_llama3_scaling():
    config = Mock(
        rope_theta=10000,
        max_position_embeddings=2048,
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 2.0,
            "low_freq_factor": 4,
            "high_freq_factor": 32,
            "original_max_position_embeddings": 2048,
        })
    weights = Mock(device=torch.device("cpu"))

    result = PositionRotaryEmbedding.static(
        config=config,
        dim=64,
        base=config.rope_theta,
        device=weights.device,
    )

    assert isinstance(result, PositionRotaryEmbedding)
    assert result.scaling_factor is None


def test_position_rotary_embedding_max_tokens_exceed_max_position_embeddings():
    config = Mock(
        rope_theta=10000,
        max_position_embeddings=4096,
        rope_scaling=None,
    )
    weights = Mock(device=torch.device("cpu"))

    with patch(
        "text_generation_server.layers.rotary._get_rope_config"
    ) as mock_get_rope_config:
        mock_get_rope_config.return_value = {"type": "dynamic", "factor": 2.0}

        result = PositionRotaryEmbedding.static(
            config=config,
            dim=64,
            base=config.rope_theta,
            device=weights.device,
        )

    assert isinstance(result, DynamicPositionRotaryEmbedding)
    assert result.scaling_factor == 2.0
    assert result.max_position_embeddings == 4096

# Test the application of the rotary embedding

def position_rotary_embedding_no_rope_config():
    head_dim = 64
    base = 10000
    max_position_embeddings = 2048
    num_heads = 16
    batch_size = 2
    seq_len = 128

    device = "cuda"
    dtype = torch.float16

    config = Mock(
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=None
    )

    # create PositionRotaryEmbedding instance
    rotary_emb = PositionRotaryEmbedding.static(
        config=config, dim=head_dim, base=base, device=device
    )

    # generate position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0)
    position_ids = position_ids.to(device).to(torch.int32).view(-1)

    # get cos and sin values for the position IDs
    cos, sin = rotary_emb.get_cos_sin(
        position_ids=position_ids,
        max_s=seq_len,
        dtype=dtype,
    )

    # create query and key tensors
    query = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device).to(dtype)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device).to(dtype)

    # clone to compare later
    original_query = query.clone()
    original_key = key.clone()

    # apply rotary embedding
    rotary_emb(query, key, cos, sin)

    # copy rotated query and key and original query and key
    q_rotated = query
    k_rotated = key
    query = original_query
    key = original_key

    assert (
        q_rotated.shape == query.shape
    ), "query shape should not change after rotation"
    assert k_rotated.shape == key.shape, "key shape should not change after rotation"
    assert not torch.allclose(q_rotated, query), "query should be modified by rotation"
    assert not torch.allclose(k_rotated, key), "key should be modified by rotation"


def position_rotary_embedding_with_dynamic_scaling():
    head_dim = 64
    base = 10000
    max_position_embeddings = 2048
    num_heads = 16
    batch_size = 2
    seq_len = 128

    device = "cuda"
    dtype = torch.float16

    config = Mock(
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        rope_scaling={"type": "dynamic", "factor": 1.0}
    )

    # create PositionRotaryEmbedding instance
    rotary_emb = PositionRotaryEmbedding.static(
        config=config, dim=head_dim, base=base, device=device
    )

    # generate position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0)
    position_ids = position_ids.to(device).to(torch.int32).view(-1)

    # get cos and sin values for the position IDs
    cos, sin = rotary_emb.get_cos_sin(
        position_ids=position_ids,
        max_s=seq_len,
        dtype=dtype,
    )

    # create query and key tensors
    query = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device).to(dtype)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device).to(dtype)

    # clone to compare later
    original_query = query.clone()
    original_key = key.clone()

    # apply rotary embedding
    rotary_emb(query, key, cos, sin)

    # copy rotated query and key and original query and key
    q_rotated = query
    k_rotated = key
    query = original_query
    key = original_key

    assert (
        q_rotated.shape == query.shape
    ), "query shape should not change after rotation"
    assert k_rotated.shape == key.shape, "key shape should not change after rotation"
    assert not torch.allclose(q_rotated, query), "query should be modified by rotation"
    assert not torch.allclose(k_rotated, key), "key should be modified by rotation"

if SYSTEM == "cuda":
    def test_position_rotary_embedding_with_dynamic_scaling():
        position_rotary_embedding_no_rope_config()

    def test_position_rotary_embedding_no_rope_config():
        position_rotary_embedding_no_rope_config()
