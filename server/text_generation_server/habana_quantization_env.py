# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import os
import sys

assert "habana_frameworks" not in sys.modules

is_quantization_enabled = os.getenv("QUANT_CONFIG", "") != ""

if is_quantization_enabled:
    os.environ.setdefault("ENABLE_EXPERIMENTAL_FLAGS", "true")
    os.environ.setdefault("USE_DEFAULT_QUANT_PARAM", "true")
    os.environ.setdefault("UPDATE_GRAPH_OUTPUT_MME", "false")
    os.environ.setdefault("ENABLE_CALC_DYNAMIC_RANGE", "false")
    os.environ.setdefault(
        "UPDATE_MME_OUTPUT_PRECISION_FILTER", "v_proj,matmul_av")
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
