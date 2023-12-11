import torch

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import modeling_auto
from typing import Optional

# Needed to properly setup habana_frameworks
import text_generation_server.habana_quantization_env as hq_env

from text_generation_server.utils.speculate import get_speculate, set_speculate
from text_generation_server.models.model import Model
from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.bloom import BLOOM
from text_generation_server.models.santacoder import SantaCoder


# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_id: str,
    revision: Optional[str],
    speculate: Optional[int],
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
) -> Model:
    if speculate is not None:
        set_speculate(speculate)
    else:
        set_speculate(0)

    config_dict, _ = PretrainedConfig.get_config_dict(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )

    use_medusa = None
    if "medusa_num_heads" in config_dict:
        use_medusa = model_id
        model_id = config_dict["base_model_name_or_path"]
        revision = "main"
        speculate_medusa = config_dict["medusa_num_heads"]
        if speculate is not None:
            if speculate > speculate_medusa:
                raise RuntimeError("Speculate is set to `{speculate}` but this medusa models only has `{speculate_medusa}` heads, please make them match")
            else:
                set_speculate(speculate)
        else:
            set_speculate(speculate_medusa)

        config_dict, _ = PretrainedConfig.get_config_dict(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        method = "medusa"
    else:
        method = "n-gram"

    speculate = get_speculate()
    if speculate > 0:
        logger.info(f"Using speculation {method} with {speculate} input ids.")

    model_type = config_dict["model_type"]

    if model_type == "gpt_bigcode":
        return SantaCoder(model_id, revision, dtype)

    if model_type == "bloom":
        return BLOOM(model_id, revision, dtype)

    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(model_id, revision, dtype)

    raise ValueError(f"Unsupported model type {model_type}")
