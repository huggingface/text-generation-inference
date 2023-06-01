import torch

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import modeling_auto
from typing import Optional

from text_generation_server.models.model import Model
from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.models.bloom import BLOOM, BLOOMSharded
from text_generation_server.models.seq2seq_lm import Seq2SeqLM
from text_generation_server.models.rw import RW
from text_generation_server.models.opt import OPT, OPTSharded
from text_generation_server.models.galactica import Galactica, GalacticaSharded
from text_generation_server.models.santacoder import SantaCoder
from text_generation_server.models.gpt_neox import GPTNeoxSharded
from text_generation_server.models.t5 import T5Sharded

try:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        is_sm75 = major == 7 and minor == 5
        is_sm8x = major == 8 and minor >= 0
        is_sm90 = major == 9 and minor == 0

        supported = is_sm75 or is_sm8x or is_sm90
        if not supported:
            raise ImportError(
                f"GPU with CUDA capability {major} {minor} is not supported"
            )

        from text_generation_server.models.flash_neox import FlashNeoX, FlashNeoXSharded
        from text_generation_server.models.flash_rw import FlashRW, FlashRWSharded
        from text_generation_server.models.flash_llama import (
            FlashLlama,
            FlashLlamaSharded,
        )
        from text_generation_server.models.flash_santacoder import (
            FlashSantacoder,
            FlashSantacoderSharded,
        )

        FLASH_ATTENTION = True
    else:
        FLASH_ATTENTION = False
except ImportError:
    logger.opt(exception=True).warning(
        "Could not import Flash Attention enabled models"
    )
    FLASH_ATTENTION = False

__all__ = [
    "Model",
    "BLOOM",
    "BLOOMSharded",
    "CausalLM",
    "FlashCausalLM",
    "Galactica",
    "GalacticaSharded",
    "GPTNeoxSharded",
    "Seq2SeqLM",
    "SantaCoder",
    "OPT",
    "OPTSharded",
    "T5Sharded",
    "get_model",
]

if FLASH_ATTENTION:
    __all__.append(FlashNeoX)
    __all__.append(FlashNeoXSharded)
    __all__.append(FlashRW)
    __all__.append(FlashRWSharded)
    __all__.append(FlashSantacoder)
    __all__.append(FlashSantacoderSharded)
    __all__.append(FlashLlama)
    __all__.append(FlashLlamaSharded)

FLASH_ATT_ERROR_MESSAGE = (
    "{} requires Flash Attention CUDA kernels to be installed.\n"
    "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
    "or install flash attention with `cd server && make install install-flash-attention`"
)

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    trust_remote_code: bool,
) -> Model:
    if "facebook/galactica" in model_id:
        if sharded:
            return GalacticaSharded(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            return Galactica(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    if model_id.startswith("bigcode/"):
        if sharded:
            if not FLASH_ATTENTION:
                raise NotImplementedError(
                    FLASH_ATT_ERROR_MESSAGE.format(f"Sharded Santacoder")
                )
            return FlashSantacoderSharded(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            santacoder_cls = FlashSantacoder if FLASH_ATTENTION else SantaCoder
            return santacoder_cls(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    config_dict, _ = PretrainedConfig.get_config_dict(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    model_type = config_dict["model_type"]

    if model_type == "gpt_bigcode":
        if sharded:
            if not FLASH_ATTENTION:
                raise NotImplementedError(
                    FLASH_ATT_ERROR_MESSAGE.format(f"Sharded Santacoder")
                )
            return FlashSantacoderSharded(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            santacoder_cls = FlashSantacoder if FLASH_ATTENTION else SantaCoder
            return santacoder_cls(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "bloom":
        if sharded:
            return BLOOMSharded(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            return BLOOM(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "gpt_neox":
        if sharded:
            neox_cls = FlashNeoXSharded if FLASH_ATTENTION else GPTNeoxSharded
            return neox_cls(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            neox_cls = FlashNeoX if FLASH_ATTENTION else CausalLM
            return neox_cls(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    if model_type in ["RefinedWeb", "RefinedWebModel"]:
        if sharded:
            if FLASH_ATTENTION:
                if config_dict.get("alibi", False) or (
                    model_type == "RefinedWebModel"
                    and config_dict.get("multi_query", True)
                ):
                    raise NotImplementedError("sharded is not supported for this model")
                return FlashRWSharded(
                    model_id,
                    revision,
                    quantize=quantize,
                    trust_remote_code=trust_remote_code,
                )
            raise NotImplementedError(
                FLASH_ATT_ERROR_MESSAGE.format(f"Sharded RefinedWeb")
            )
        else:
            if FLASH_ATTENTION and not config_dict.get("alibi", False):
                return FlashRW(
                    model_id,
                    revision,
                    quantize=quantize,
                    trust_remote_code=trust_remote_code,
                )
            else:
                return RW(
                    model_id,
                    revision,
                    quantize=quantize,
                    trust_remote_code=trust_remote_code,
                )

    if model_type == "llama":
        if sharded:
            if FLASH_ATTENTION:
                return FlashLlamaSharded(
                    model_id,
                    revision,
                    quantize=quantize,
                    trust_remote_code=trust_remote_code,
                )
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format(f"Sharded Llama"))
        else:
            llama_cls = FlashLlama if FLASH_ATTENTION else CausalLM
            return llama_cls(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "opt":
        if sharded:
            return OPTSharded(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            return OPT(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "t5":
        if sharded:
            return T5Sharded(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            return Seq2SeqLM(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    if sharded:
        raise ValueError("sharded is not supported for AutoModel")

    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(
            model_id, revision, quantize=quantize, trust_remote_code=trust_remote_code
        )
    if model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
        return Seq2SeqLM(
            model_id, revision, quantize=quantize, trust_remote_code=trust_remote_code
        )

    auto_map = config_dict.get("auto_map", None)
    if trust_remote_code and auto_map is not None:
        if "AutoModelForCausalLM" in auto_map.keys():
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        if "AutoModelForSeq2SeqLM" in auto_map.keys():
            return Seq2SeqLM(
                model_id,
                revision,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )

    raise ValueError(f"Unsupported model type {model_type}")
