# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import os
import habana_frameworks.torch as htorch

quant_config = os.getenv("QUANT_CONFIG", "")
is_quantization_enabled = quant_config != ""

if is_quantization_enabled:
    os.environ.setdefault("ENABLE_EXPERIMENTAL_FLAGS", "true")
    os.environ.setdefault("USE_DEFAULT_QUANT_PARAM", "true")
    os.environ.setdefault("UPDATE_GRAPH_OUTPUT_MME", "false")
    os.environ.setdefault("ENABLE_CALC_DYNAMIC_RANGE", "false")
    os.environ.setdefault("UPDATE_MME_OUTPUT_PRECISION_FILTER", "v_proj,matmul_av")
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")


def patch_scoped_linear_all_reduce(model):
    from deepspeed.module_inject.layers import LinearAllreduce
    from optimum.habana.transformers.models.modeling_all_models import ScopedLinearAllReduce

    for name, module in model.named_children():
        if type(module) is LinearAllreduce:
            SL = ScopedLinearAllReduce(mod=module)
            setattr(model, name, SL)
        patch_scoped_linear_all_reduce(module)


def setup_quantization(model):
    if is_quantization_enabled:
        htorch.core.quantization._mark_params_as_const(model)
        htorch.core.quantization._check_params_as_const(model)
        htorch.core.hpu_initialize(model)
    return model


def prepare_model_for_quantization(model):
    if is_quantization_enabled:
        if model.config.model_type in ["llama", "falcon", "qwen2", "starcoder2", "gemma"]:
            patch_scoped_linear_all_reduce(model)
        from neural_compressor.torch.quantization import FP8Config, convert

        config = FP8Config.from_json_file(quant_config)
        model = convert(model, config)
    return model
