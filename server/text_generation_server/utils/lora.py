import json
from text_generation_server.utils import (
    hub,
)
import os


class LoraConfig:
    def __init__(
        self,
        alpha_pattern=None,
        auto_mapping=None,
        base_model_name_or_path="",
        bias="none",
        fan_in_fan_out=False,
        inference_mode=True,
        init_lora_weights=True,
        layer_replication=None,
        layers_pattern=None,
        layers_to_transform=None,
        loftq_config=None,
        lora_alpha=16,
        lora_dropout=0.1,
        megatron_config=None,
        megatron_core="megatron.core",
        modules_to_save=None,
        peft_type="LORA",
        r=8,
        rank_pattern=None,
        revision=None,
        target_modules=None,
        task_type="CAUSAL_LM",
        use_dora=False,
        use_rslora=False,
        config_path=None,
    ):
        self.alpha_pattern = alpha_pattern or {}
        self.auto_mapping = auto_mapping
        self.base_model_name_or_path = base_model_name_or_path
        self.bias = bias
        self.fan_in_fan_out = fan_in_fan_out
        self.inference_mode = inference_mode
        self.init_lora_weights = init_lora_weights
        self.layer_replication = layer_replication
        self.layers_pattern = layers_pattern
        self.layers_to_transform = layers_to_transform
        self.loftq_config = loftq_config or {}
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.megatron_config = megatron_config
        self.megatron_core = megatron_core
        self.modules_to_save = modules_to_save
        self.peft_type = peft_type
        self.r = r
        self.rank_pattern = rank_pattern or {}
        self.revision = revision
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.task_type = task_type
        self.use_dora = use_dora
        self.use_rslora = use_rslora
        self.config_path = config_path

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            json_data = json.load(f)
        return cls(**json_data, config_path=filename)

    # TODO: support fetching the model from the hub if it's not in the cache
    @classmethod
    def from_pretrained(cls, adapter_id, revision=None):
        d = hub._get_cached_revision_directory(adapter_id, revision)
        filename = os.path.join(d, "adapter_config.json")
        return cls.from_file(filename)
