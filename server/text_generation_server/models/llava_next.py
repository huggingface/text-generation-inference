import torch
import torch.distributed

from typing import List, Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoProcessor,
)
from text_generation_server.models.custom_modeling.llava_next import (
    LlavaNextForConditionalGeneration,
)

# from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from text_generation_server.models.vlm_causal_lm import VlmCausalLM
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


class LlavaNext(VlmCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            # 9b seems to work correctly enough in float16, but 80b seems
            # to be really saturating for f16.
            dtype = torch.float16 if dtype is None else dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype
        self.device, self.dtype = device, dtype

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        config.quantize = quantize
        config.use_medusa = use_medusa
        config.vision_config.quantize = quantize

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device=device,
            dtype=dtype,
            process_group=self.process_group,
        )

        model = LlavaNextForConditionalGeneration(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(VlmCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
