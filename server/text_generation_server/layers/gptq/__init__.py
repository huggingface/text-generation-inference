import os
import torch
from text_generation_server.utils.import_utils import (
    SYSTEM,
)

try:
    major, _minor = torch.cuda.get_device_capability()
except Exception:
    major = 1

HAS_EXLLAMA = False
CAN_EXLLAMA = major >= 8 or SYSTEM == "rocm"
V2 = os.getenv("EXLLAMA_VERSION", "2") == "2"
if os.getenv("DISABLE_EXLLAMA") == "True":
    HAS_EXLLAMA = False
elif CAN_EXLLAMA:
    try:
        if V2:
            from text_generation_server.layers.gptq.exllamav2 import (
                QuantLinear as ExllamaQuantLinear,
                create_exllama_buffers,
                set_device,
            )

            HAS_EXLLAMA = "2"
        else:
            from text_generation_server.layers.gptq.exllama import (
                Ex4bitLinear as ExllamaQuantLinear,
                create_exllama_buffers,
                set_device,
            )

            HAS_EXLLAMA = "1"

    except ImportError:
        pass

from text_generation_server.layers.gptq.quant_linear import QuantLinear
