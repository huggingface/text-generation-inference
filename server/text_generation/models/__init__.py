from text_generation.models.model import Model
from text_generation.models.bloom import BLOOM, BLOOMSharded

__all__ = ["Model", "BLOOM", "BLOOMSharded"]


def get_model(model_name: str, sharded: bool, quantize: bool) -> Model:
    if model_name.startswith("bigscience/bloom"):
        if sharded:
            return BLOOMSharded(model_name, quantize)
        else:
            if quantize:
                raise ValueError("quantization is not supported for non-sharded BLOOM")
            return BLOOM(model_name)
    else:
        raise ValueError(f"model {model_name} is not supported yet")
