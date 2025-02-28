import json
import os
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import hf_hub_download
from text_generation_server.layers.marlin.gptq import can_use_gptq_marlin
from text_generation_server.utils.weights import (
    DefaultWeightsLoader,
    WeightsLoader,
)


# TODO: Split this config to have a single config type per quant method
@dataclass
class _QuantizerConfig:
    bits: int
    checkpoint_format: Optional[str]
    desc_act: bool
    groupsize: int
    quant_method: str
    sym: bool


@dataclass
class _FP8QuantizerConfig:
    activation_scale_ub: float


# We should probably do this with Pytantic JSON deserialization,
# but for now we'll stay close to the old _set_gptq_params.
def _get_quantizer_config(model_id, revision):
    bits = 4
    groupsize = -1
    quant_method = "gptq"
    checkpoint_format = None
    sym = False
    desc_act = False

    filename = "config.json"
    try:
        if os.path.exists(os.path.join(model_id, filename)):
            filename = os.path.join(model_id, filename)
        else:
            filename = hf_hub_download(model_id, filename=filename, revision=revision)
        with open(filename, "r") as f:
            data = json.load(f)

        # FP8 config
        if data["quantization_config"]["quant_method"] == "fbgemm_fp8":
            return _FP8QuantizerConfig(
                activation_scale_ub=data["quantization_config"]["activation_scale_ub"]
            )

        if "zero_point" in data["quantization_config"]:
            sym = not data["quantization_config"]["zero_point"]
            quant_method = "awq"
        elif "sym" in data["quantization_config"]:
            sym = data["quantization_config"]["sym"]

        bits = data["quantization_config"]["bits"]
        groupsize = data["quantization_config"]["group_size"]
        # Order is important here, desc_act is missing on some real models
        quant_method = data["quantization_config"]["quant_method"]
        checkpoint_format = data["quantization_config"].get("checkpoint_format")
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

            if "zero_point" in data:
                sym = not data["zero_point"]
                quant_method = "awq"
            elif "sym" in data:
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

        # TODO: improve check once we have one config type per quantize value
        if not isinstance(quantizer_config, _QuantizerConfig):
            raise ValueError(
                f"Quantize is set to `{quantize}` but received a `{quantizer_config.__class__.__name__}` config."
            )

        if can_use_gptq_marlin(
            bits=quantizer_config.bits,
            groupsize=quantizer_config.groupsize,
            quant_method=quantizer_config.quant_method,
            quantize=quantize,
            sym=quantizer_config.sym,
        ):
            from text_generation_server.layers.marlin import GPTQMarlinWeightsLoader

            return GPTQMarlinWeightsLoader(
                bits=quantizer_config.bits,
                desc_act=quantizer_config.desc_act,
                groupsize=quantizer_config.groupsize,
                quant_method=quantizer_config.quant_method,
                quantize=quantize,
                sym=quantizer_config.sym,
            )
        else:
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
    elif quantize == "marlin":
        from text_generation_server.layers.marlin import MarlinWeightsLoader

        # TODO: improve check once we have one config type per quantize value
        if not isinstance(quantizer_config, _QuantizerConfig):
            raise ValueError(
                f"Quantize is set to `{quantize}` but received a `{quantizer_config.__class__.__name__}` config."
            )

        return MarlinWeightsLoader(
            bits=quantizer_config.bits,
            is_marlin_24=quantizer_config.checkpoint_format == "marlin_24",
        )
    elif quantize == "fp8" or quantize is None:
        from text_generation_server.layers.fp8 import HybridFP8UnquantLoader

        # Since the default for the quantize config is _QuantizerConfig,
        # we need to add this check to not get an attribute error
        activation_scale_ub = None
        if isinstance(quantizer_config, _FP8QuantizerConfig):
            activation_scale_ub = quantizer_config.activation_scale_ub

        return HybridFP8UnquantLoader(activation_scale_ub, to_fp8=quantize == "fp8")
    else:
        raise ValueError(f"Unknown quantization method: {quantize}")
