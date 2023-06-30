import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from typing import Optional
from huggingface_hub import hf_hub_download
import json

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_mpt_modeling import (
    MPTForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class MPTSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16
        else:
            raise NotImplementedError("FlashMPT is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        filename = hf_hub_download(model_id, revision=revision, filename="config.json")
        with open(filename, "r") as f:
            config = json.load(f)
        config = PretrainedConfig(**config)
        config.quantize = quantize
        # config = AutoConfig.from_pretrained(
        #     # model_id, revision=revision, trust_remote_code=trust_remote_code
        #     model_id, revision=revision, trust_remote_code=False
        # )

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)

        config.quantize = quantize
        model = MPTForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
