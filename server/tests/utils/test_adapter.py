import pytest
from unittest.mock import Mock
from text_generation_server.utils.adapter import get_attn_weights, get_mlp_weights


def test_get_attn_weights():
    # create a mock layer
    mock_layer = Mock()
    mock_layer.self_attn.query_key_value = Mock()
    mock_layer.self_attn.o_proj = Mock()

    # call the function
    result = get_attn_weights(2, mock_layer)

    # assert the result
    expected = {
        (2, "q_proj"): (
            "model.layers.2.self_attn.q_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "k_proj"): (
            "model.layers.2.self_attn.k_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "v_proj"): (
            "model.layers.2.self_attn.v_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "o_proj"): ("model.layers.2.self_attn.o_proj", mock_layer.self_attn.o_proj),
    }
    assert result == expected


def test_get_mlp_weights_with_gate_up_proj():
    # create a mock layer with gate_up_proj
    mock_layer = Mock()
    mock_layer.mlp.gate_up_proj = Mock()
    mock_layer.mlp.down_proj = Mock()

    # call the function
    result = get_mlp_weights(3, mock_layer)

    # assert the result
    expected = {
        (3, "gate_proj"): ("model.layers.3.mlp.gate_proj", mock_layer.mlp.gate_up_proj),
        (3, "up_proj"): ("model.layers.3.mlp.up_proj", mock_layer.mlp.gate_up_proj),
        (3, "down_proj"): ("model.layers.3.mlp.down_proj", mock_layer.mlp.down_proj),
    }
    assert result == expected


def test_get_mlp_weights_without_gate_up_proj():
    # create a mock layer without gate_up_proj
    mock_layer = Mock()
    mock_layer.mlp = Mock(spec=[])

    # call the function
    result = get_mlp_weights(1, mock_layer)

    # assert the result
    assert result == {}


@pytest.mark.parametrize("layer_index", [0, 1, 5])
def test_get_attn_weights_different_layers(layer_index):
    mock_layer = Mock()
    mock_layer.self_attn.query_key_value = Mock()
    mock_layer.self_attn.o_proj = Mock()

    result = get_attn_weights(layer_index, mock_layer)

    for k in ["q", "k", "v"]:
        assert (layer_index, f"{k}_proj") in result
        assert (
            result[(layer_index, f"{k}_proj")][0]
            == f"model.layers.{layer_index}.self_attn.{k}_proj"
        )

    assert (layer_index, "o_proj") in result
    assert (
        result[(layer_index, "o_proj")][0]
        == f"model.layers.{layer_index}.self_attn.o_proj"
    )


@pytest.mark.parametrize("layer_index", [0, 1, 5])
def test_get_mlp_weights_different_layers(layer_index):
    mock_layer = Mock()
    mock_layer.mlp.gate_up_proj = Mock()
    mock_layer.mlp.down_proj = Mock()

    result = get_mlp_weights(layer_index, mock_layer)

    for k in ["gate", "up", "down"]:
        assert (layer_index, f"{k}_proj") in result
        assert (
            result[(layer_index, f"{k}_proj")][0]
            == f"model.layers.{layer_index}.mlp.{k}_proj"
        )


def test_get_attn_weights_llama_compatibility():
    mock_layer = Mock()
    mock_layer.self_attn.query_key_value = Mock()
    mock_layer.self_attn.o_proj = Mock()

    result = get_attn_weights(2, mock_layer)

    expected = {
        (2, "q_proj"): (
            "model.layers.2.self_attn.q_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "k_proj"): (
            "model.layers.2.self_attn.k_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "v_proj"): (
            "model.layers.2.self_attn.v_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "o_proj"): ("model.layers.2.self_attn.o_proj", mock_layer.self_attn.o_proj),
    }
    assert result == expected


def test_get_mlp_weights_llama_compatibility():
    mock_layer = Mock()
    mock_layer.mlp.gate_up_proj = Mock()
    mock_layer.mlp.down_proj = Mock()

    result = get_mlp_weights(3, mock_layer)

    expected = {
        (3, "gate_proj"): ("model.layers.3.mlp.gate_proj", mock_layer.mlp.gate_up_proj),
        (3, "up_proj"): ("model.layers.3.mlp.up_proj", mock_layer.mlp.gate_up_proj),
        (3, "down_proj"): ("model.layers.3.mlp.down_proj", mock_layer.mlp.down_proj),
    }
    assert result == expected


def test_get_attn_weights_gemma_compatibility():
    mock_layer = Mock()
    mock_layer.self_attn.query_key_value = Mock()
    mock_layer.self_attn.o_proj = Mock()

    result = get_attn_weights(2, mock_layer)

    expected = {
        (2, "q_proj"): (
            "model.layers.2.self_attn.q_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "k_proj"): (
            "model.layers.2.self_attn.k_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "v_proj"): (
            "model.layers.2.self_attn.v_proj",
            mock_layer.self_attn.query_key_value,
        ),
        (2, "o_proj"): ("model.layers.2.self_attn.o_proj", mock_layer.self_attn.o_proj),
    }
    assert result == expected


def test_get_mlp_weights_gemma_compatibility():
    mock_layer = Mock()
    mock_layer.mlp.gate_proj = Mock()
    mock_layer.mlp.up_proj = Mock()
    mock_layer.mlp.down_proj = Mock()

    # ensure that the mock_layer.mlp.gate_up_proj attribute does not exist.
    # This is necessary because the use of `Mock` automatically creates any
    # attributes that are accessed, even if they don't exist in the actual
    # implementation. If `gate_up_proj` were created, `get_mlp_weights` might
    # follow the wrong execution path and return an incorrect result.
    del mock_layer.mlp.gate_up_proj

    result = get_mlp_weights(3, mock_layer)

    expected = {
        (3, "gate_proj"): ("model.layers.3.mlp.gate_proj", mock_layer.mlp.gate_proj),
        (3, "up_proj"): ("model.layers.3.mlp.up_proj", mock_layer.mlp.up_proj),
        (3, "down_proj"): ("model.layers.3.mlp.down_proj", mock_layer.mlp.down_proj),
    }
    assert result == expected
