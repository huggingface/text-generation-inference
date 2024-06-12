import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoConfig, AutoTokenizer
from typing import Optional

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_phi_modeling import (
    FlashPhiForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashPhi(FlashCausalLM):
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
            raise NotImplementedError("FlashPhi is only available on GPU")

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

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = FlashPhiForCausalLM(config, weights)
        if speculator:
            from text_generation_server.utils.medusa import MedusaModel
            from huggingface_hub import hf_hub_download
            import json
            import os
            from pathlib import Path

            is_local_model = (
                Path(speculator).exists() and Path(speculator).is_dir()
            ) or os.getenv("WEIGHTS_CACHE_OVERRIDE", None) is not None

            if not is_local_model:
                medusa_config = hf_hub_download(
                    speculator, revision=revision, filename="config.json"
                )
                medusa_head = hf_hub_download(
                    speculator, revision=revision, filename="medusa_lm_head.pt"
                )
            else:
                medusa_config = str(Path(speculator) / "config.json")
                medusa_head = str(Path(speculator) / "medusa_lm_head.pt")

            with open(medusa_config, "r") as f:
                config = json.load(f)
            medusa_sf = medusa_head[: -len(".pt")] + ".safetensors"
            weights = Weights(
                [medusa_sf], device, dtype, process_group=self.process_group
            )
            lm_head = model.lm_head
            model.lm_head = MedusaModel(config, weights, lm_head)

        torch.distributed.barrier(group=self.process_group)
        super(FlashPhi, self).__init__(
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
