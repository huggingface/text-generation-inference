import torch
import torch.distributed

from opentelemetry import trace
from typing import Optional
from transformers.models.gemma import GemmaTokenizerFast
from transformers import AutoConfig, PretrainedConfig

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_gemma_modeling import (
    FlashGemmaForCausalLM,
    GemmaConfig,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class VisionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        model_type: str = "siglip_vision_model",
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        num_image_tokens: int = 256,
        patch_size: int = 14,
        projection_dim: int = 2048,
        projector_hidden_act: str = "gelu_fast",
        vision_use_head: bool = False,
        vocab_size: int = 257152,
        quantize: Optional[str] = None,
        image_size: int = 224,
        layer_norm_eps: float = 1e-06,
        attention_dropout: float = 0.0,
        hidden_act: str = "gelu_pytorch_tanh",
        num_channels: int = 3,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_image_tokens = num_image_tokens
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.projector_hidden_act = projector_hidden_act
        self.vision_use_head = vision_use_head
        self.vocab_size = vocab_size
        self.quantize = quantize
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.num_channels = num_channels


class BaseFlashGemma(FlashCausalLM):
    def __init__(
        self,
        model_cls,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        prefix: Optional[str] = None,
        config_cls=AutoConfig,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.bfloat16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashGemma is only available on GPU")

        tokenizer = GemmaTokenizerFast.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
            use_fast=True,
            from_slow=False,
        )

        config = config_cls.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        is_vlm = hasattr(config, "vision_config")

        config.quantize = quantize
        config.speculator = speculator

        if is_vlm:
            config.intermediate_size = config.text_config.get("intermediate_size")
            config.num_attention_heads = config.text_config.get("num_attention_heads")
            config.num_hidden_layers = config.text_config.get("num_hidden_layers")
            config.num_key_value_heads = config.text_config.get("num_key_value_heads")

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = model_cls(prefix, config, weights)

        torch.distributed.barrier(group=self.process_group)

        if is_vlm:
            num_layers = config.num_hidden_layers
            num_kv_heads = config.num_key_value_heads
            head_size = config.intermediate_size
        else:
            num_layers = len(model.model.layers)
            num_kv_heads = model.model.num_key_value_heads
            head_size = model.model.head_size

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )


class FlashGemma(BaseFlashGemma):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        super(FlashGemma, self).__init__(
            model_cls=FlashGemmaForCausalLM,
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            use_medusa=use_medusa,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            prefix=None,
        )
