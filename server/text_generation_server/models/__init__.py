import torch

from transformers import AutoConfig
from typing import Optional

from text_generation_server.models.model import Model
from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.bloom import BLOOM, BLOOMSharded
from text_generation_server.models.seq2seq_lm import Seq2SeqLM
from text_generation_server.models.galactica import Galactica, GalacticaSharded
from text_generation_server.models.santacoder import SantaCoder
from text_generation_server.models.gpt_neox import GPTNeoxSharded
from text_generation_server.models.t5 import T5Sharded

__all__ = [
    "Model",
    "BLOOM",
    "BLOOMSharded",
    "CausalLM",
    "Galactica",
    "GalacticaSharded",
    "GPTNeoxSharded",
    "Seq2SeqLM",
    "SantaCoder",
    "T5Sharded",
    "get_model",
]

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_id: str, revision: Optional[str], sharded: bool, quantize: bool
) -> Model:
    if "facebook/galactica" in model_id:
        if sharded:
            return GalacticaSharded(model_id, revision, quantize=quantize)
        else:
            return Galactica(model_id, revision, quantize=quantize)

    if "santacoder" in model_id:
        return SantaCoder(model_id, revision, quantize)

    config = AutoConfig.from_pretrained(model_id, revision=revision)

    if config.model_type == "bloom":
        if sharded:
            return BLOOMSharded(model_id, revision, quantize=quantize)
        else:
            return BLOOM(model_id, revision, quantize=quantize)

    if config.model_type == "gpt_neox":
        if sharded:
            return GPTNeoxSharded(model_id, revision, quantize=quantize)
        else:
            return CausalLM(model_id, revision, quantize=quantize)

    if config.model_type == "t5":
        if sharded:
            return T5Sharded(model_id, revision, quantize=quantize)
        else:
            return Seq2SeqLM(model_id, revision, quantize=quantize)

    if sharded:
        raise ValueError("sharded is not supported for AutoModel")
    try:
        return CausalLM(model_id, revision, quantize=quantize)
    except Exception:
        return Seq2SeqLM(model_id, revision, quantize=quantize)
