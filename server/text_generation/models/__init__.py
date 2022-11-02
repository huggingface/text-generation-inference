from text_generation.models.model import Model
from text_generation.models.bloom import BLOOMSharded

__all__ = ["Model", "BLOOMSharded"]


def get_model(model_name: str, sharded: bool, quantize: bool) -> Model:

    if model_name.startswith("bigscience/bloom"):
        if sharded:
            return BLOOMSharded(model_name, quantize)
        else:
            if quantize:
                raise ValueError("quantization is not supported for non-sharded BLOOM")
            return Model(model_name)
    else:
        if sharded:
            raise ValueError("sharded is only supported for BLOOM models")
        if quantize:
            raise ValueError("Quantization is only supported for BLOOM models")

        return Model(model_name)
