import torch
import torch.distributed

from typing import Optional


from text_generation_server.models.custom_modeling.idefics_config import IdeficsConfig
from text_generation_server.models.custom_modeling.idefics_processing import (
    IdeficsProcessor,
)
from transformers import LlamaTokenizerFast
from text_generation_server.models.custom_modeling.idefics_modeling import (
    IdeficsForVisionText2Text,
)
from text_generation_server.models.idefics_causal_lm import IdeficsCausalLM
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from text_generation_server.utils.quantization import get_loader

from text_generation_server.utils.import_utils import SYSTEM


class IDEFICSSharded(IdeficsCausalLM):
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
            # 9b seems to work correctly enough in float16, but 80b seems
            # to be really saturating for f16.
            dtype = torch.float16 if dtype is None else dtype
        elif SYSTEM == "ipex":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device(f"xpu:{rank}")
                dtype = torch.float16 if dtype is None else dtype
            else:
                device = torch.device("cpu")
                # Float16 doesn't exist on target.
                dtype = torch.bfloat16 if dtype is None else dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype
        self.device, self.dtype = device, dtype

        config = IdeficsConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        config.quantize = quantize
        config.speculator = speculator
        config.vision_config.quantize = quantize

        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        self.processor = IdeficsProcessor.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        weights_loader = get_loader(
            quantize=quantize, model_id=model_id, revision=revision
        )
        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device=device,
            dtype=dtype,
            process_group=self.process_group,
            weights_loader=weights_loader,
        )

        model = IdeficsForVisionText2Text(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(IdeficsCausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
