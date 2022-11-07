from text_generation.models.model import Model
from text_generation.models.causal_lm import CausalLM
from text_generation.models.bloom import BLOOMSharded
from text_generation.models.seq2seq_lm import Seq2SeqLM

__all__ = ["Model", "BLOOMSharded", "CausalLM", "Seq2SeqLM"]


def get_model(model_name: str, sharded: bool, quantize: bool) -> Model:
    if model_name.startswith("bigscience/bloom"):
        if sharded:
            return BLOOMSharded(model_name, quantize=quantize)
        else:
            return CausalLM(model_name, quantize=quantize)
    else:
        if sharded:
            raise ValueError("sharded is not supported for AutoModel")
        try:
            return CausalLM(model_name, quantize=quantize)
        except Exception as e:
            return Seq2SeqLM(model_name, quantize=quantize)
