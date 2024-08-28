# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import os

quant_config = os.getenv("QUANT_CONFIG", "")
is_quantization_enabled = quant_config != ""

if is_quantization_enabled:
    os.environ.setdefault("ENABLE_EXPERIMENTAL_FLAGS", "true")
    os.environ.setdefault("USE_DEFAULT_QUANT_PARAM", "true")
    os.environ.setdefault("UPDATE_GRAPH_OUTPUT_MME", "false")
    os.environ.setdefault("ENABLE_CALC_DYNAMIC_RANGE", "false")
    os.environ.setdefault(
        "UPDATE_MME_OUTPUT_PRECISION_FILTER", "v_proj,matmul_av")
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")


def prepare_model_for_quantization(model):
    if is_quantization_enabled:
        if os.getenv("USE_INC", "1") != "0":
            from neural_compressor.torch.quantization import FP8Config, convert
            config = FP8Config.from_json_file(quant_config)
            model = convert(model, config)
        else:
            import habana_quantization_toolkit
            habana_quantization_toolkit.prep_model(model)
        return model
