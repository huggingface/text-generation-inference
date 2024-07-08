import torch
from typing import Optional, Tuple, Dict, List

from text_generation_server.models import FlashCausalLM


ADAPTER_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
ROW_PARALLEL = {"o_proj", "down_proj", "lm_head"}


class FlashMistral(FlashCausalLM):
    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"

        # This accounts for VLMs (e.g. LlavaNext, Idefics2)
        # that have a language_model inside of the larger model.
        if hasattr(self.model, "text_model"):
            _model = self.model.text_model
        else:
            _model = self.model

        for i, layer in enumerate(_model.model.layers):
            layer_weights[(i, "q_proj")] = (
                f"{prefix}.{i}.self_attn.q_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, "k_proj")] = (
                f"{prefix}.{i}.self_attn.k_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, "v_proj")] = (
                f"{prefix}.{i}.self_attn.v_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, "o_proj")] = (
                f"{prefix}.{i}.self_attn.o_proj",
                layer.self_attn.o_proj,
            )

            # TODO: this is a hack to avoid the gate_proj for
            # FlashStarcoder2 that doesnt have these layers
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_up_proj"):
                layer_weights[(i, "gate_proj")] = (
                    f"{prefix}.{i}.mlp.gate_proj",
                    layer.mlp.gate_up_proj,
                )
                layer_weights[(i, "up_proj")] = (
                    f"{prefix}.{i}.mlp.up_proj",
                    layer.mlp.gate_up_proj,
                )
                layer_weights[(i, "down_proj")] = (
                    f"{prefix}.{i}.mlp.down_proj",
                    layer.mlp.down_proj,
                )

        layer_weights[(0, "lm_head")] = ("lm_head", _model.lm_head)
        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return ["q_proj", "v_proj"]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == "lm_head" else len(self.model.model.layers)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
