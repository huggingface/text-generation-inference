import torch
import torch.distributed

from transformers import AutoConfig, AutoTokenizer
from typing import Optional, List, Tuple

from text_generation_server.models import CausalLM
from text_generation_server.models.custom_modeling.phi_modeling import (
    PhiConfig,
    PhiForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


class Phi(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, _rank, _world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        config = PhiConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        config.quantize = quantize
        config.use_medusa = use_medusa
        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        model = PhiForCausalLM(config, weights)
        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )
