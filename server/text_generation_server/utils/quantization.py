import json
import os
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import hf_hub_download
from text_generation_server.utils.weights import (
    DefaultWeightsLoader,
    UnquantizedWeight,
    WeightsLoader,
)


@dataclass
class _QuantizerConfig:
    bits: int
    checkpoint_format: Optional[str]
    desc_act: bool
    groupsize: int
    quant_method: str
    sym: bool


# We should probably do this with Pytantic JSON deserialization,
# but for now we'll stay close to the old _set_gptq_params.
def _get_quantizer_config(model_id, revision):
    bits = 4
    groupsize = -1
    quant_method = "gptq"
    checkpoint_format = None
    sym = True
    desc_act = False

    filename = "config.json"
    try:
        if os.path.exists(os.path.join(model_id, filename)):
            filename = os.path.join(model_id, filename)
        else:
            filename = hf_hub_download(model_id, filename=filename, revision=revision)
        with open(filename, "r") as f:
            data = json.load(f)
        bits = data["quantization_config"]["bits"]
        groupsize = data["quantization_config"]["group_size"]
        # Order is important here, desc_act is missing on some real models
        quant_method = data["quantization_config"]["quant_method"]
        checkpoint_format = data["quantization_config"].get("checkpoint_format")
        sym = data["quantization_config"]["sym"]
        desc_act = data["quantization_config"]["desc_act"]
    except Exception:
        filename = "quantize_config.json"
        try:
            if os.path.exists(os.path.join(model_id, filename)):
                filename = os.path.join(model_id, filename)
            else:
                filename = hf_hub_download(
                    model_id, filename=filename, revision=revision
                )
            with open(filename, "r") as f:
                data = json.load(f)
            bits = data["bits"]
            groupsize = data["group_size"]
            sym = data["sym"]
            desc_act = data["desc_act"]
            if "version" in data and data["version"] == "GEMM":
                quant_method = "awq"
        except Exception:
            filename = "quant_config.json"
            try:
                if os.path.exists(os.path.join(model_id, filename)):
                    filename = os.path.join(model_id, filename)
                else:
                    filename = hf_hub_download(
                        model_id, filename=filename, revision=revision
                    )
                with open(filename, "r") as f:
                    data = json.load(f)
                bits = data["w_bit"]
                groupsize = data["q_group_size"]
                desc_act = data["desc_act"]
                if "version" in data and data["version"] == "GEMM":
                    quant_method = "awq"
            except Exception:
                pass

    return _QuantizerConfig(
        bits=bits,
        groupsize=groupsize,
        quant_method=quant_method,
        checkpoint_format=checkpoint_format,
        sym=sym,
        desc_act=desc_act,
    )


def get_loader(
    quantize: Optional[str], model_id: str, revision: Optional[str]
) -> WeightsLoader:
    quantizer_config = _get_quantizer_config(model_id, revision)
    if quantize in {"awq", "gptq"}:
        from text_generation_server.layers.gptq import GPTQWeightsLoader

        return GPTQWeightsLoader(
            bits=quantizer_config.bits,
            desc_act=quantizer_config.desc_act,
            groupsize=quantizer_config.groupsize,
            quant_method=quantizer_config.quant_method,
            quantize=quantize,
            sym=quantizer_config.sym,
        )
    elif quantize == "bitsandbytes":
        from text_generation_server.layers.bnb import BNBWeight

        return DefaultWeightsLoader(BNBWeight)
    elif quantize == "bitsandbytes-fp4":
        from text_generation_server.layers.bnb import BNBFP4Weight

        return DefaultWeightsLoader(BNBFP4Weight)
    elif quantize == "bitsandbytes-nf4":
        from text_generation_server.layers.bnb import BNBNF4Weight

        return DefaultWeightsLoader(BNBNF4Weight)
    elif quantize == "eetq":
        from text_generation_server.layers.eetq import EETQWeight

        return DefaultWeightsLoader(EETQWeight)
    elif quantize == "exl2":
        from text_generation_server.layers.exl2 import Exl2WeightsLoader

        return Exl2WeightsLoader()
    elif quantize == "fp8":
        from text_generation_server.layers.fp8 import Fp8Weight

        return DefaultWeightsLoader(Fp8Weight)
    elif quantize == "marlin":
        from text_generation_server.layers.marlin import MarlinWeightsLoader

        return MarlinWeightsLoader(
            bits=quantizer_config.bits,
            is_marlin_24=quantizer_config.checkpoint_format == "marlin_24",
        )
    elif quantize is None:
        return DefaultWeightsLoader(UnquantizedWeight)
    else:
        raise ValueError(f"Unknown quantization method: {quantize}")
