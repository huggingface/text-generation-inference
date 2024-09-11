import pytest
from unittest.mock import Mock
from text_generation_server.utils.adapter import (
    get_attn_weights,
    get_mlp_weights,
    parse_lora_adapters,
    AdapterInfo,
)


def test_parse_lora_adapters_empty():
    assert parse_lora_adapters(None) == []
    assert parse_lora_adapters("") == []


def test_parse_lora_adapters_single():
    result = parse_lora_adapters("adapter1")
    assert result == [AdapterInfo(id="adapter1", path=None, revision=None)]


def test_parse_lora_adapters_with_path():
    result = parse_lora_adapters("adapter1=path/to/adapter1")
    assert result == [
        AdapterInfo(id="adapter1", path="path/to/adapter1", revision=None)
    ]


def test_parse_lora_adapters_with_path_and_revision():
    result = parse_lora_adapters("adapter1=path/to/adapter1@main")
    assert result == [
        AdapterInfo(id="adapter1", path="path/to/adapter1", revision="main")
    ]


def test_parse_lora_adapters_multiple():
    result = parse_lora_adapters(
        "adapter1,adapter2=path/to/adapter2,adapter3=path/to/adapter3@dev"
    )
    assert result == [
        AdapterInfo(id="adapter1", path=None, revision=None),
        AdapterInfo(id="adapter2", path="path/to/adapter2", revision=None),
        AdapterInfo(id="adapter3", path="path/to/adapter3", revision="dev"),
    ]


def test_parse_lora_adapters_invalid_format():
    try:
        parse_lora_adapters("adapter1,invalid=format=test,adapter3")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Invalid LoRA adapter format: invalid=format=test"


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
        (2, "qkv_proj"): (
            "model.layers.2.self_attn.qkv_proj",
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
        (2, "qkv_proj"): (
            "model.layers.2.self_attn.qkv_proj",
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
        (2, "qkv_proj"): (
            "model.layers.2.self_attn.qkv_proj",
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
