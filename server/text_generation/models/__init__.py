import torch

from text_generation.models.model import Model
from text_generation.models.causal_lm import CausalLM
from text_generation.models.bloom import BLOOM, BLOOMSharded
from text_generation.models.seq2seq_lm import Seq2SeqLM
from text_generation.models.galactica import Galactica, GalacticaSharded
from text_generation.models.santacoder import SantaCoder

__all__ = [
    "Model",
    "BLOOM",
    "BLOOMSharded",
    "CausalLM",
    "Seq2SeqLM",
    "SantaCoder",
    "get_model",
]

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def get_model(model_name: str, sharded: bool, quantize: bool) -> Model:
    if model_name.startswith("bigscience/bloom"):
        if sharded:
            return BLOOMSharded(model_name, quantize=quantize)
        else:
            return BLOOM(model_name, quantize=quantize)
    elif model_name.startswith("facebook/galactica"):
        if sharded:
            return GalacticaSharded(model_name, quantize=quantize)
        else:
            return Galactica(model_name, quantize=quantize)
    elif "santacoder" in model_name:
        return SantaCoder(model_name, quantize)
    else:
        if sharded:
            raise ValueError("sharded is not supported for AutoModel")
        try:
            return CausalLM(model_name, quantize=quantize)
        except Exception:
            return Seq2SeqLM(model_name, quantize=quantize)
