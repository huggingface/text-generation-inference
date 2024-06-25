import os
import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from typing import Optional, Tuple, Dict, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
    hub,
)

tracer = trace.get_tracer(__name__)

from text_generation_server.utils.import_utils import SYSTEM

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


class FlashLlama(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        lora_adapter_ids: Optional[list] = [],
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        elif SYSTEM == "ipex":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device(f"xpu:{rank}")
                dtype = torch.float16 if dtype is None else dtype
            else:
                device = torch.device("cpu")
                dtype = torch.bfloat16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            if isinstance(generation_config.eos_token_id, (list, set)):
                # TODO Huge hack
                tokenizer._eos_token_ids = set(generation_config.eos_token_id)
        except Exception:
            pass

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.speculator = speculator

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["awq", "exl2", "gptq", "marlin"]:
            weights._set_gptq_params(model_id, revision)

        prefix = ""
        model = FlashLlamaForCausalLM(prefix, config, weights)
        torch.distributed.barrier(group=self.process_group)
        super(FlashLlama, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"

        # This accounts for VLMs (e.g. LlavaNext, Idefics2)
        # that have a language_model inside of the larger model.
        if hasattr(self.model, "language_model"):
            _model = self.model.language_model
        elif hasattr(self.model, "text_model"):
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
