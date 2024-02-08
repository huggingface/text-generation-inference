import torch

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import modeling_auto
from typing import Optional

from text_generation_server.utils.speculate import get_speculate, set_speculate
from text_generation_server.models.model import Model
from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.models.bloom import BLOOMSharded
from text_generation_server.models.mpt import MPTSharded
from text_generation_server.models.seq2seq_lm import Seq2SeqLM
from text_generation_server.models.rw import RW
from text_generation_server.models.opt import OPTSharded
from text_generation_server.models.galactica import GalacticaSharded
from text_generation_server.models.santacoder import SantaCoder
from text_generation_server.models.t5 import T5Sharded
from text_generation_server.models.gpt_neox import GPTNeoxSharded
from text_generation_server.models.phi import Phi

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)

__all__ = [
    "Model",
    "BLOOMSharded",
    "CausalLM",
    "FlashCausalLM",
    "GalacticaSharded",
    "Seq2SeqLM",
    "SantaCoder",
    "OPTSharded",
    "T5Sharded",
    "get_model",
]

FLASH_ATT_ERROR_MESSAGE = "{} requires Flash Attention enabled models."

FLASH_ATTENTION = True
try:
    from text_generation_server.models.flash_rw import FlashRWSharded
    from text_generation_server.models.flash_neox import FlashNeoXSharded
    from text_generation_server.models.flash_llama import (
        FlashLlama,
    )
    from text_generation_server.models.flash_santacoder import (
        FlashSantacoderSharded,
    )
    from text_generation_server.models.idefics import IDEFICSSharded
    from text_generation_server.models.flash_mistral import FlashMistral
    from text_generation_server.models.flash_mixtral import FlashMixtral
    from text_generation_server.models.flash_phi import FlashPhi
    from text_generation_server.utils.flash_attn import HAS_FLASH_ATTN_V2_CUDA

except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False
    HAS_FLASH_ATTN_V2_CUDA = False

if FLASH_ATTENTION:
    __all__.append(FlashNeoXSharded)
    __all__.append(FlashRWSharded)
    __all__.append(FlashSantacoderSharded)
    __all__.append(FlashLlama)
    __all__.append(IDEFICSSharded)
    __all__.append(FlashMistral)
    __all__.append(FlashMixtral)
    __all__.append(FlashPhi)

MAMBA_AVAILABLE = True
try:
    from text_generation_server.models.mamba import Mamba
except ImportError as e:
    logger.warning(f"Could not import Mamba: {e}")
    MAMBA_AVAILABLE = False

if MAMBA_AVAILABLE:
    __all__.append(Mamba)


def get_model(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[str],
    trust_remote_code: bool,
) -> Model:
    if dtype is None:
        # Keep it as default for now and let
        # every model resolve their own default dtype.
        dtype = None
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    if speculate is not None:
        set_speculate(speculate)
    else:
        set_speculate(0)

    if "facebook/galactica" in model_id:
        return GalacticaSharded(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_id.startswith("bigcode/"):
        if FLASH_ATTENTION:
            return FlashSantacoderSharded(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(
                FLASH_ATT_ERROR_MESSAGE.format("Sharded Santacoder")
            )
        else:
            return SantaCoder(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

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
                raise RuntimeError(
                    "Speculate is set to `{speculate}` but this medusa models only has `{speculate_medusa}` heads, please make them match"
                )
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

    model_type = config_dict.get("model_type", None)
    if model_type is None:
        # TODO: fix how we determine model type for Mamba
        if "ssm_cfg" in config_dict:
            # *only happens in Mamba case
            model_type = "ssm"
        else:
            raise RuntimeError(
                f"Could not determine model type for {model_id} revision {revision}"
            )

    if model_type == "ssm":
        return Mamba(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "gpt_bigcode":
        if FLASH_ATTENTION:
            return FlashSantacoderSharded(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(
                FLASH_ATT_ERROR_MESSAGE.format("Sharded Santacoder")
            )
        else:
            return SantaCoder(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "bloom":
        return BLOOMSharded(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    elif model_type == "mpt":
        return MPTSharded(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    elif model_type == "gpt_neox":
        if FLASH_ATTENTION:
            return FlashNeoXSharded(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            return GPTNeoxSharded(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    elif model_type == "phi":
        if FLASH_ATTENTION:
            return FlashPhi(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                use_medusa=use_medusa,
            )
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    elif model_type == "phi-msft":
        if FLASH_ATTENTION:
            raise NotImplementedError(
                "Legacy phi-msft is not supported with Flash Attention"
            )
        else:
            return Phi(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    elif model_type == "llama" or model_type == "baichuan":
        if FLASH_ATTENTION:
            return FlashLlama(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                use_medusa=use_medusa,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Llama"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type in ["RefinedWeb", "RefinedWebModel", "falcon"]:
        if sharded:
            if FLASH_ATTENTION:
                if config_dict.get("alibi", False):
                    raise NotImplementedError("sharded is not supported for this model")
                return FlashRWSharded(
                    model_id,
                    revision,
                    quantize=quantize,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format(f"Sharded Falcon"))
        else:
            if FLASH_ATTENTION and not config_dict.get("alibi", False):
                return FlashRWSharded(
                    model_id,
                    revision,
                    quantize=quantize,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            else:
                return RW(
                    model_id,
                    revision,
                    quantize=quantize,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )

    if model_type == "mistral":
        sliding_window = config_dict.get("sliding_window", -1)
        if (
            (sliding_window is None or sliding_window == -1) and FLASH_ATTENTION
        ) or HAS_FLASH_ATTN_V2_CUDA:
            return FlashMistral(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "mixtral":
        sliding_window = config_dict.get("sliding_window", -1)
        if (
            (sliding_window is None or sliding_window == -1) and FLASH_ATTENTION
        ) or HAS_FLASH_ATTN_V2_CUDA:
            return FlashMixtral(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "opt":
        return OPTSharded(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "t5":
        return T5Sharded(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if model_type == "idefics":
        if FLASH_ATTENTION:
            return IDEFICSSharded(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Idefics"))

    if sharded:
        raise NotImplementedError("sharded is not supported for AutoModel")
    if quantize == "gptq":
        raise NotImplementedError(
            "gptq quantization is not supported for AutoModel, you can try to quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
        )
    if quantize == "awq":
        raise NotImplementedError("awq quantization is not supported for AutoModel")
    elif (quantize == "bitsandbytes-fp4") or (quantize == "bitsandbytes-nf4"):
        raise NotImplementedError("4bit quantization is not supported for AutoModel")
    elif quantize == "eetq":
        raise NotImplementedError("Eetq quantization is not supported for AutoModel")
    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
        return Seq2SeqLM(
            model_id,
            revision,
            quantize=quantize,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    auto_map = config_dict.get("auto_map", None)
    if trust_remote_code and auto_map is not None:
        if "AutoModelForCausalLM" in auto_map.keys():
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        if "AutoModelForSeq2SeqLM" in auto_map.keys():
            return Seq2SeqLM(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    raise ValueError(f"Unsupported model type {model_type}")
