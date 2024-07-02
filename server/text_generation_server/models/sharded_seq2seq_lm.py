import torch
import torch.distributed

from typing import List, Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoConfig,
)

from text_generation_server.models import Seq2SeqLM
from text_generation_server.models.custom_modeling.t5_modeling import (
    T5ForConditionalGeneration,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


class ShardedSeq2SeqLM(Seq2SeqLM):
    def __init__(
        self,
        model_id: str,
        model_class,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        config_class=AutoConfig,
        tokenizer_class=AutoTokenizer,
        aliases=None,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        config = config_class.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        config.quantize = quantize
        config.speculator = speculator

        tokenizer = tokenizer_class.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        tokenizer.bos_token_id = config.decoder_start_token_id

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device=device,
            dtype=dtype,
            process_group=self.process_group,
            aliases=aliases,
        )

        model = model_class(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(Seq2SeqLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
