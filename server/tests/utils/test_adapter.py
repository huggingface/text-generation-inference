import torch
from peft import LoraConfig

from text_generation_server.utils.adapter import merge_adapter_weights


def test_merge_adapter_weights():
    W_0 = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    model_weights = {
        "model.layers.10.self_attn.q_proj.weight": W_0
    }
    
    A = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    B = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    adapter_weights = {
        "base_model.model.model.layers.10.self_attn.q_proj.lora_A.weight": A,
        "base_model.model.model.layers.10.self_attn.q_proj.lora_B.weight": B
    }

    W_expected = torch.tensor([
        [ 5.5000,  8.0000, 10.5000],
        [13.5000, 18.0000, 22.5000],
        [21.5000, 28.0000, 34.5000]
    ])
    adapter_config = LoraConfig(r=2, lora_alpha=1, fan_in_fan_out=False)
    merged_weights, processed_adapter_weight_names = merge_adapter_weights(model_weights, adapter_weights, adapter_config)

    assert len(merged_weights) == 1
    assert merged_weights["model.layers.10.self_attn.q_proj.weight"].equal(W_expected)
    
    assert len(processed_adapter_weight_names) == 2
    assert "base_model.model.model.layers.10.self_attn.q_proj.lora_A.weight" in processed_adapter_weight_names
    assert "base_model.model.model.layers.10.self_attn.q_proj.lora_B.weight" in processed_adapter_weight_names