import math

import torch

from typing import Optional

from transformers.models.gpt2 import GPT2TokenizerFast

from text_generation_server.models.cache_manager import BLOCK_SIZE
from text_generation_server.models.flash_mistral import (
    BaseFlashMistral,
    set_sliding_window,
)
from text_generation_server.models.custom_modeling.flash_starcoder2_modeling import (
    Starcoder2Config,
    FlashStarcoder2ForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


# Starcoder2 has the same base as Mistral
class FlashStarcoder2(BaseFlashMistral):
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
            raise NotImplementedError("FlashStarcoder2 is only available on GPU")

        tokenizer = GPT2TokenizerFast.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = Starcoder2Config.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.speculator = speculator

        # Set context windows
        if config.sliding_window is not None:
            set_sliding_window(
                config.sliding_window, math.ceil(config.sliding_window / BLOCK_SIZE)
            )

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = FlashStarcoder2ForCausalLM(config, weights)

        self.cuda_graphs = {}

        torch.distributed.barrier(group=self.process_group)
        super(BaseFlashMistral, self).__init__(
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
