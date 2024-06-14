import math

import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoTokenizer, AutoConfig
from typing import Optional

from text_generation_server.models.flash_mistral import (
    BaseFlashMistral,
    set_sliding_window,
)
from text_generation_server.models.custom_modeling.flash_qwen2_modeling import (
    Qwen2ForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashQwen2(BaseFlashMistral):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashQwen2 is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.speculator = speculator

        # Set context windows
        if config.sliding_window is not None:
            set_sliding_window(config.sliding_window)

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq", "marlin"]:
            weights._set_gptq_params(model_id, revision)

        model = Qwen2ForCausalLM(config, weights)

        self.cuda_graphs = {}

        torch.distributed.barrier(group=self.process_group)
        super(BaseFlashMistral, self).__init__(
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
            sliding_window=config.sliding_window,
        )
