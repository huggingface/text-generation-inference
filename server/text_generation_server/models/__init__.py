import torch
import enum
import os

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import modeling_auto
from huggingface_hub import hf_hub_download, HfApi
from typing import Optional
from pathlib import Path

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
    from text_generation_server.models.flash_gpt2 import FlashGPT2
    from text_generation_server.models.flash_neox import FlashNeoXSharded
    from text_generation_server.models.flash_llama import (
        FlashLlama,
    )
    from text_generation_server.models.flash_qwen2 import (
        FlashQwen2,
    )
    from text_generation_server.models.flash_cohere import (
        FlashCohere,
    )
    from text_generation_server.models.flash_gemma import (
        FlashGemma,
    )
    from text_generation_server.models.pali_gemma import (
        PaliGemma,
    )
    from text_generation_server.models.flash_santacoder import (
        FlashSantacoderSharded,
    )
    from text_generation_server.models.idefics import IDEFICSSharded
    from text_generation_server.models.llava_next import LlavaNext
    from text_generation_server.models.idefics2 import Idefics2
    from text_generation_server.models.flash_mistral import FlashMistral
    from text_generation_server.models.flash_mixtral import FlashMixtral
    from text_generation_server.models.flash_phi import FlashPhi
    from text_generation_server.models.flash_starcoder2 import FlashStarcoder2
    from text_generation_server.models.flash_dbrx import FlashDbrx
    from text_generation_server.layers.attention import SUPPORTS_WINDOWING
except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    SUPPORTS_WINDOWING = False
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashGPT2)
    __all__.append(FlashNeoXSharded)
    __all__.append(FlashRWSharded)
    __all__.append(FlashSantacoderSharded)
    __all__.append(FlashLlama)
    __all__.append(IDEFICSSharded)
    __all__.append(FlashMistral)
    __all__.append(FlashMixtral)
    __all__.append(FlashDbrx)
    __all__.append(FlashPhi)
    __all__.append(FlashQwen2)
    __all__.append(FlashStarcoder2)
    __all__.append(FlashGemma)
    __all__.append(FlashCohere)

MAMBA_AVAILABLE = True
try:
    from text_generation_server.models.mamba import Mamba
except ImportError as e:
    logger.warning(f"Could not import Mamba: {e}")
    MAMBA_AVAILABLE = False

if MAMBA_AVAILABLE:
    __all__.append(Mamba)


class ModelType(enum.Enum):
    IDEFICS2 = {
        "type": "idefics2",
        "name": "Idefics 2",
        "url": "https://huggingface.co/HuggingFaceM4/idefics2-8b",
        "multimodal": True,
    }
    LLAVA_NEXT = {
        "type": "llava_next",
        "name": "Llava Next (1.6)",
        "url": "https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf",
        "multimodal": True,
    }
    LLAMA = {
        "type": "llama",
        "name": "Llama",
        "url": "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
    }
    PHI3 = {
        "type": "phi3",
        "name": "Phi 3",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
    }
    GEMMA = {
        "type": "gemma",
        "name": "Gemma",
        "url": "https://huggingface.co/google/gemma-7b",
    }
    COHERE = {
        "type": "cohere",
        "name": "Cohere",
        "url": "https://huggingface.co/CohereForAI/c4ai-command-r-plus",
    }
    DBRX = {
        "type": "dbrx",
        "name": "Dbrx",
        "url": "https://huggingface.co/databricks/dbrx-instruct",
    }
    MAMBA = {
        "type": "ssm",
        "name": "Mamba",
        "url": "https://huggingface.co/state-spaces/mamba-2.8b-slimpj",
    }
    MISTRAL = {
        "type": "mistral",
        "name": "Mistral",
        "url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
    }
    MIXTRAL = {
        "type": "mixtral",
        "name": "Mixtral",
        "url": "https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1",
    }
    GPT_BIGCODE = {
        "type": "gpt_bigcode",
        "name": "Gpt Bigcode",
        "url": "https://huggingface.co/bigcode/gpt_bigcode-santacoder",
    }
    PHI = {
        "type": "phi",
        "name": "Phi",
        "url": "https://huggingface.co/microsoft/phi-1_5",
    }
    BAICHUAN = {
        "type": "baichuan",
        "name": "Baichuan",
        "url": "https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat",
    }
    FALCON = {
        "type": "falcon",
        "name": "Falcon",
        "url": "https://huggingface.co/tiiuae/falcon-7b-instruct",
    }
    STARCODER2 = {
        "type": "starcoder2",
        "name": "StarCoder 2",
        "url": "https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1",
    }
    QWEN2 = {
        "type": "qwen2",
        "name": "Qwen 2",
        "url": "https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1",
    }
    OPT = {
        "type": "opt",
        "name": "Opt",
        "url": "https://huggingface.co/facebook/opt-6.7b",
    }
    T5 = {
        "type": "t5",
        "name": "T5",
        "url": "https://huggingface.co/google/flan-t5-xxl",
    }
    GALACTICA = {
        "type": "galactica",
        "name": "Galactica",
        "url": "https://huggingface.co/facebook/galactica-120b",
    }
    SANTACODER = {
        "type": "santacoder",
        "name": "SantaCoder",
        "url": "https://huggingface.co/bigcode/santacoder",
    }
    BLOOM = {
        "type": "bloom",
        "name": "Bloom",
        "url": "https://huggingface.co/bigscience/bloom-560m",
    }
    MPT = {
        "type": "mpt",
        "name": "Mpt",
        "url": "https://huggingface.co/mosaicml/mpt-7b-instruct",
    }
    GPT2 = {
        "type": "gpt2",
        "name": "Gpt2",
        "url": "https://huggingface.co/openai-community/gpt2",
    }
    GPT_NEOX = {
        "type": "gpt_neox",
        "name": "Gpt Neox",
        "url": "https://huggingface.co/EleutherAI/gpt-neox-20b",
    }
    IDEFICS = {
        "type": "idefics",
        "name": "Idefics",
        "url": "https://huggingface.co/HuggingFaceM4/idefics-9b",
        "multimodal": True,
    }


__GLOBALS = locals()
for data in ModelType:
    __GLOBALS[data.name] = data.value["type"]


def get_model(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[str],
    trust_remote_code: bool,
) -> Model:
    global FLASH_ATTENTION
    if dtype is None:
        if quantize in ["awq", "exl2", "gptq"]:
            # These quantizers only work with float16 params.
            dtype = torch.float16
        else:
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

    config_dict, _ = PretrainedConfig.get_config_dict(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    model_type = config_dict.get("model_type", None)

    speculator = None
    if "medusa_num_heads" in config_dict:
        medusa_model_id = model_id
        medusa_revision = revision
        model_id = config_dict["base_model_name_or_path"]
        revision = "main"
        speculate_medusa = config_dict["medusa_num_heads"]
        if speculate is not None:
            if speculate > speculate_medusa:
                raise RuntimeError(
                    f"Speculate is set to `{speculate}` but this medusa models only has `{speculate_medusa}` heads, please make them match"
                )
            else:
                set_speculate(speculate)
        else:
            set_speculate(speculate_medusa)

        config_dict, _ = PretrainedConfig.get_config_dict(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        # Reload model type from parent.
        model_type = config_dict.get("model_type", None)
        is_local = Path(medusa_model_id).exists()
        if not is_local:
            medusa_config = hf_hub_download(
                medusa_model_id, revision=medusa_revision, filename="config.json"
            )
            hf_hub_download(
                medusa_model_id,
                revision=medusa_revision,
                filename="medusa_lm_head.safetensors",
            )
            speculator = {
                "path": Path(medusa_config).parent,
                "model_paths": ["medusa_lm_head.safetensors"],
            }
        else:
            speculator = {
                "path": Path(medusa_model_id),
                "model_paths": ["medusa_lm_head.safetensors"],
            }

        method = "medusa"
    elif model_type == "mlp_speculator":
        mlp_model_id = model_id
        mlp_revision = revision
        model_id = config_dict["base_model_name_or_path"]
        revision = "main"
        speculate_mlp = config_dict["n_predict"]
        if speculate is not None:
            if speculate > speculate_mlp:
                raise RuntimeError(
                    f"Speculate is set to `{speculate}` but this mlp_speculator models only has `{speculate_mlp}` heads, please make them match"
                )
            else:
                set_speculate(speculate)
        else:
            set_speculate(speculate_mlp)

        config_dict, _ = PretrainedConfig.get_config_dict(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        # Reload model type from parent.
        model_type = config_dict.get("model_type", None)
        is_local = Path(mlp_model_id).exists()
        extension = ".safetensors"
        if not is_local:
            mlp_speculator_config = hf_hub_download(
                mlp_model_id, revision=mlp_revision, filename="config.json"
            )
            api = HfApi()
            info = api.model_info(mlp_model_id, revision=mlp_revision)
            filenames = [
                s.rfilename
                for s in info.siblings
                if s.rfilename.endswith(extension)
                and len(s.rfilename.split("/")) == 1
                and "arguments" not in s.rfilename
                and "args" not in s.rfilename
                and "training" not in s.rfilename
            ]
            for filename in filenames:
                hf_hub_download(
                    mlp_model_id,
                    revision=mlp_revision,
                    filename=filename,
                )
            speculator = {
                "path": Path(mlp_speculator_config).parent,
                "model_paths": filenames,
            }
        else:
            speculator = Path(mlp_model_id)
            filenames = [p for p in os.listdir(speculator) if p.endswith(extension)]
            speculator = {"path": speculator, "model_paths": filenames}
        method = "mlp_speculator"
    else:
        method = "n-gram"

    speculate = get_speculate()
    if speculate > 0:
        logger.info(f"Using speculation {method} with {speculate} input ids.")

    if model_type is None:
        # TODO: fix how we determine model type for Mamba
        if "ssm_cfg" in config_dict:
            # *only happens in Mamba case
            model_type = "ssm"
        else:
            raise RuntimeError(
                f"Could not determine model type for {model_id} revision {revision}"
            )
    quantization_config = config_dict.get("quantization_config", None)
    if quantization_config is not None and quantize is None:
        method = quantization_config.get("quant_method", None)
        if method in {"gptq", "awq", "exl2"}:
            logger.info(f"Auto selecting quantization method {method}")
            quantize = method
        else:
            logger.info(f"Unknown quantization method {method}")

    if quantize == "exl2" and sharded:
        raise RuntimeError(
            "Sharding is currently not supported with `exl2` quantization"
        )
    sliding_window = config_dict.get("sliding_window", -1)
    if sliding_window != -1 and not SUPPORTS_WINDOWING:
        logger.warning(
            f"Flash attention is available, but doesn't support windowing which is required by model {model_id}"
        )
        FLASH_ATTENTION = False

    if model_type == MAMBA:
        return Mamba(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_id.startswith("facebook/galactica"):
        return GalacticaSharded(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if (
        model_type == GPT_BIGCODE
        or model_type == GPT2
        and model_id.startswith("bigcode/")
    ):
        if FLASH_ATTENTION:
            return FlashSantacoderSharded(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
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
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == BLOOM:
        return BLOOMSharded(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    elif model_type == MPT:
        return MPTSharded(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    elif model_type == GPT2:
        if FLASH_ATTENTION:
            try:
                return FlashGPT2(
                    model_id,
                    revision,
                    quantize=quantize,
                    speculator=speculator,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            except RuntimeError as e:
                # Lots of legacy models with various weight names.
                logger.warning(f"Couldn't load flash gpt2 variant: {e}")
                return CausalLM(
                    model_id,
                    revision,
                    quantize=quantize,
                    speculator=speculator,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded GPT-2"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
    elif model_type == GPT_NEOX:
        if FLASH_ATTENTION:
            return FlashNeoXSharded(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            return GPTNeoxSharded(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    elif model_type == PHI:
        if FLASH_ATTENTION:
            return FlashPhi(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
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
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    elif model_type == LLAMA or model_type == BAICHUAN or model_type == PHI3:
        if FLASH_ATTENTION:
            return FlashLlama(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Llama"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
    if model_type == GEMMA:
        if FLASH_ATTENTION:
            return FlashGemma(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Gemma"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == COHERE:
        if FLASH_ATTENTION:
            return FlashCohere(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Cohere"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == DBRX:
        if FLASH_ATTENTION:
            return FlashDbrx(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded DBRX"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type in ["RefinedWeb", "RefinedWebModel", FALCON]:
        if sharded:
            if FLASH_ATTENTION:
                if config_dict.get("alibi", False):
                    raise NotImplementedError("sharded is not supported for this model")
                return FlashRWSharded(
                    model_id,
                    revision,
                    quantize=quantize,
                    speculator=speculator,
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
                    speculator=speculator,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            else:
                return RW(
                    model_id,
                    revision,
                    quantize=quantize,
                    speculator=speculator,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )

    if model_type == MISTRAL:
        sliding_window = config_dict.get("sliding_window", -1)
        if FLASH_ATTENTION:
            return FlashMistral(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Mistral"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == MIXTRAL:
        sliding_window = config_dict.get("sliding_window", -1)
        if FLASH_ATTENTION:
            return FlashMixtral(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Mixtral"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == STARCODER2:
        sliding_window = config_dict.get("sliding_window", -1)
        if FLASH_ATTENTION:
            return FlashStarcoder2(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(
                FLASH_ATT_ERROR_MESSAGE.format("Sharded Starcoder2")
            )
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == QWEN2:
        sliding_window = config_dict.get("sliding_window", -1)
        if (sliding_window is None or sliding_window != -1) and SUPPORTS_WINDOWING:
            return FlashQwen2(
                model_id,
                revision,
                quantize=quantize,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Qwen2"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == OPT:
        return OPTSharded(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == T5:
        return T5Sharded(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if model_type == IDEFICS:
        if FLASH_ATTENTION:
            return IDEFICSSharded(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Idefics"))
    if model_type == IDEFICS2:
        if FLASH_ATTENTION:
            return Idefics2(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Idefics"))
    if model_type == "paligemma":
        if FLASH_ATTENTION:
            return PaliGemma(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Idefics"))

    if model_type == LLAVA_NEXT:
        if FLASH_ATTENTION:
            return LlavaNext(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("LlavaNext"))

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
    elif quantize == "exl2":
        raise NotImplementedError("exl2 quantization is not supported for AutoModel")
    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
        return Seq2SeqLM(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
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
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        if "AutoModelForSeq2SeqLM" in auto_map.keys():
            return Seq2SeqLM(
                model_id,
                revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    raise ValueError(f"Unsupported model type {model_type}")
