# ruff: noqa: F821
# the above line disables the `undefined-name` rule for the model type variables
import torch
import os

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import modeling_auto
from huggingface_hub import hf_hub_download, HfApi
from typing import Optional
from pathlib import Path
from typing import List, Dict
import enum

# Needed to properly setup habana_frameworks

from text_generation_server.utils.speculate import get_speculate, set_speculate
from text_generation_server.models.model import Model
from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.bloom import BLOOM
from text_generation_server.models.starcoder import StarCoder
from text_generation_server.models.custom_modeling.flash_phi_moe_modeling import (
    PhiMoEConfig,
)

from text_generation_server.utils.adapter import (
    AdapterParameters,
    build_layer_weight_lookup,
    load_and_merge_adapters,
    AdapterInfo,
)
from text_generation_server.adapters.lora import LoraWeights

from text_generation_server.utils.log import log_master
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

__all__ = [
    "Model",
    "CausalLM",
    "Seq2SeqLM",
    "get_model_with_lora_adapters",
]
from text_generation_server.models.globals import ATTENTION

FLASH_ATT_ERROR_MESSAGE = "{} requires Flash Attention enabled models."

FLASH_ATTENTION = False
if ATTENTION == "paged":
    FLASH_ATTENTION = True

try:
    from text_generation_server.models.flash_causal_lm import FlashCausalLM
    from text_generation_server.models.flash_vlm_causal_lm import FlashVlmCausalLM
    from text_generation_server.models.mllama_causal_lm import FlashMllamaCausalLM
    from text_generation_server.models.custom_modeling.flash_deepseek_v2_modeling import (
        FlashDeepseekV2ForCausalLM,
        DeepseekV2Config,
    )
    from text_generation_server.models.custom_modeling.flash_deepseek_v3_modeling import (
        FlashDeepseekV3ForCausalLM,
        DeepseekV3Config,
    )
    from text_generation_server.models.custom_modeling.flash_llama_modeling import (
        FlashLlamaForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_cohere_modeling import (
        FlashCohereForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_gemma_modeling import (
        FlashGemmaForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_gemma2_modeling import (
        FlashGemma2ForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_dbrx_modeling import (
        FlashDbrxForCausalLM,
        DbrxConfig,
    )
    from text_generation_server.models.custom_modeling.flash_rw_modeling import (
        RWConfig,
        FlashRWForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_neox_modeling import (
        FlashGPTNeoXForCausalLM,
    )
    from text_generation_server.models.pali_gemma import (
        PaliGemmaBatch,
    )
    from text_generation_server.models.custom_modeling.flash_pali_gemma_modeling import (
        PaliGemmaForConditionalGeneration,
    )
    from text_generation_server.models.custom_modeling.flash_phi_modeling import (
        FlashPhiForCausalLM,
    )
    from text_generation_server.models.mllama_causal_lm import FlashMllamaCausalLMBatch
    from text_generation_server.models.custom_modeling.flash_mllama import (
        FlashMllamaForConditionalGeneration,
    )
    from text_generation_server.models.custom_modeling.flash_llava_next import (
        FlashLlavaNextForConditionalGeneration,
    )

    from text_generation_server.models.custom_modeling.flash_santacoder_modeling import (
        FlashSantacoderForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_starcoder2_modeling import (
        FlashStarcoder2ForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_qwen2_modeling import (
        Qwen2ForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_mistral_modeling import (
        FlashMistralForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_mixtral_modeling import (
        FlashMixtralForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_gpt2_modeling import (
        FlashGPT2ForCausalLM,
    )
    from text_generation_server.models.custom_modeling.flash_gptj_modeling import (
        FlashGPTJForCausalLM,
    )
    from text_generation_server.models.custom_modeling.idefics2 import (
        Idefics2ForConditionalGeneration,
    )
    from text_generation_server.models.custom_modeling.idefics3 import (
        Idefics3ForConditionalGeneration,
    )
    from text_generation_server.models.custom_modeling.qwen2_vl import (
        Qwen2VLForConditionalGeneration,
    )
    from text_generation_server.models.custom_modeling.qwen2_5_vl import (
        Qwen2_5VLForConditionalGeneration,
        Qwen2_5_VLConfig,
        Qwen2_5_VLProcessor,
    )
    from text_generation_server.layers.attention import SUPPORTS_WINDOWING
except ImportError as e:
    log_master(logger.warning, f"Could not import Flash Attention enabled models: {e}")
    SUPPORTS_WINDOWING = False
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashCausalLM)


class ModelType(enum.Enum):
    DEEPSEEK_V2 = {
        "type": "deepseek_v2",
        "name": "Deepseek V2",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V2",
    }
    DEEPSEEK_V3 = {
        "type": "deepseek_v3",
        "name": "Deepseek V3",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V3",
    }
    IDEFICS2 = {
        "type": "idefics2",
        "name": "Idefics 2",
        "url": "https://huggingface.co/HuggingFaceM4/idefics2-8b",
        "multimodal": True,
    }
    IDEFICS3 = {
        "type": "idefics3",
        "name": "Idefics 3",
        "url": "https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3",
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
        "url": "https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f",
    }
    PHI3 = {
        "type": "phi3",
        "name": "Phi 3",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
    }
    GRANITE = {
        "type": "granite",
        "name": "Granite",
        "url": "https://huggingface.co/ibm-granite/granite-3.0-8b-instruct",
    }
    GEMMA = {
        "type": "gemma",
        "name": "Gemma",
        "url": "https://huggingface.co/google/gemma-7b",
    }
    PALIGEMMA = {
        "type": "paligemma",
        "name": "PaliGemma",
        "url": "https://huggingface.co/google/paligemma-3b-pt-224",
    }
    GEMMA2 = {
        "type": "gemma2",
        "name": "Gemma2",
        "url": "https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315",
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
        "type": "mamba",
        "name": "Mamba",
        "url": "https://huggingface.co/state-spaces/mamba-2.8b-slimpj",
    }
    MISTRAL = {
        "type": "mistral",
        "name": "Mistral",
        "url": "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
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
    PHI_MOE = {
        "type": "phimoe",
        "name": "PhiMoe",
        "url": "https://huggingface.co/microsoft/Phi-3.5-MoE-instruct",
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
        "url": "https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f",
    }
    QWEN2_VL = {
        "type": "qwen2_vl",
        "name": "Qwen 2 VL",
        "url": "https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d",
    }
    QWEN2_5_VL = {
        "type": "qwen2_5_vl",
        "name": "Qwen 2.5 VL",
        "url": "https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e",
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
    GPTJ = {
        "type": "gptj",
        "name": "Gptj",
        "url": "https://huggingface.co/EleutherAI/gpt-j-6b",
    }
    MLLAMA = {
        "type": "mllama",
        "name": "Mllama",
        "url": "https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct",
        "multimodal": True,
    }


__GLOBALS = locals()
for data in ModelType:
    __GLOBALS[data.name] = data.value["type"]

SDP_ON_BF16 = int(os.environ.get("SDP_ON_BF16", 0))
# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_id: str,
    lora_adapter_ids: Optional[List[str]],
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    max_input_tokens: int,
) -> Model:
    global FLASH_ATTENTION

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
            speculator_dir_path = Path(mlp_speculator_config).parent
            # if these are downloaded, they get converted to safetensors
            filenames.extend(
                [p for p in os.listdir(speculator_dir_path) if p.endswith(extension)]
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

    model_type = config_dict["model_type"]

    kv_cache_dtype = dtype

    if FLASH_ATTENTION:
        if model_type == DEEPSEEK_V2:
            head_size = max(
                config_dict.get("qk_nope_dim", 128)
                + config_dict.get("qk_rope_dim", 64),
                config_dict.get("v_head_dim", 128),
            )
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashDeepseekV2ForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                default_dtype=torch.bfloat16,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                config_class=DeepseekV2Config,
                head_size=head_size,
            )
        elif model_type == DEEPSEEK_V3:
            head_size = max(
                config_dict.get("qk_nope_dim", 128)
                + config_dict.get("qk_rope_dim", 64),
                config_dict.get("v_head_dim", 128),
            )
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashDeepseekV3ForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                default_dtype=torch.bfloat16,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                config_class=DeepseekV3Config,
                head_size=head_size,
            )

        elif (
            model_type == GPT_BIGCODE
            or model_type == GPT2
            and model_id.startswith("bigcode/")
        ):
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashSantacoderForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                aliases={"transformer.wte.weight": ["lm_head.weight"]},
                num_kv_heads=1,
            )
        elif model_type == GPT2:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashGPT2ForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == GPTJ:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashGPTJForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == GPT_NEOX:
            from text_generation_server.models.custom_modeling.flash_neox_modeling import (
                GPTNeoXConfig,
            )

            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashGPTNeoXForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                config_class=GPTNeoXConfig,
            )
        elif model_type == PHI:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashPhiForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == PHI_MOE:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashLlamaForCausalLM,
                config_class=PhiMoEConfig,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == LLAMA or model_type == PHI3 or model_type == GRANITE:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashLlamaForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == BAICHUAN:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashLlamaForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == GEMMA:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashGemmaForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                # Works better for these models
                default_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == GEMMA2:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashGemma2ForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                # Works better for these models
                default_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == COHERE:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashCohereForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == DBRX:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashDbrxForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                # Dbrx works better in bfloat16.
                default_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                config_class=DbrxConfig,
            )
        elif (
            model_type in ["RefinedWeb", "RefinedWebModel", FALCON]
            and not sharded
            and not config_dict.get("alibi", False)
        ):
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashRWForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                aliases={
                    "lm_head.weight": ["transformer.word_embeddings.weight"],
                    "transformer.word_embeddings.weight": ["lm_head.weight"],
                },
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                config_class=RWConfig,
            )
        elif model_type == MISTRAL:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashMistralForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == MIXTRAL:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashMixtralForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == STARCODER2:
            return FlashCausalLM(
                model_id=model_id,
                model_class=FlashStarcoder2ForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == QWEN2:
            return FlashCausalLM(
                model_id=model_id,
                model_class=Qwen2ForCausalLM,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == QWEN2_VL:
            return FlashVlmCausalLM(
                model_id=model_id,
                model_class=Qwen2VLForConditionalGeneration,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                default_dtype=torch.bfloat16,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == QWEN2_5_VL:
            return FlashVlmCausalLM(
                model_id=model_id,
                model_class=Qwen2_5VLForConditionalGeneration,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                default_dtype=torch.bfloat16,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                config_class=Qwen2_5_VLConfig,
                processor_class=Qwen2_5_VLProcessor,
            )
        elif model_type == MLLAMA:
            return FlashMllamaCausalLM(
                model_id=model_id,
                model_class=FlashMllamaForConditionalGeneration,
                batch_class=FlashMllamaCausalLMBatch,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                default_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
            )
        elif model_type == IDEFICS2:
            return FlashVlmCausalLM(
                model_id=model_id,
                model_class=Idefics2ForConditionalGeneration,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                # XXX: Extremely important to cap resolution in order to limit
                # VRAM usage.
                processor_kwargs={"size": {"longest_edge": 448, "shortest_edge": 378}},
            )
        elif model_type == IDEFICS3:
            return FlashVlmCausalLM(
                model_id=model_id,
                model_class=Idefics3ForConditionalGeneration,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                default_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                # XXX: Extremely important to cap resolution in order to limit
                # VRAM usage.
                processor_kwargs={"size": {"longest_edge": 1456}},
            )
        elif model_type == PALIGEMMA:
            return FlashVlmCausalLM(
                model_id=model_id,
                model_class=PaliGemmaForConditionalGeneration,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                # Works better for these models
                default_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
                lora_adapter_ids=lora_adapter_ids,
                batch_class=PaliGemmaBatch,
            )
        elif model_type == LLAVA_NEXT:
            return FlashVlmCausalLM(
                model_class=FlashLlavaNextForConditionalGeneration,
                model_id=model_id,
                revision=revision,
                quantize=quantize,
                speculator=speculator,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                trust_remote_code=trust_remote_code,
            )

    from text_generation_server.models.vlm_causal_lm import VlmCausalLM
    from text_generation_server.models.custom_modeling.mllama import (
        MllamaForConditionalGeneration,
    )
    from text_generation_server.models.custom_modeling.llava_next import (
        LlavaNextForConditionalGeneration,
    )

    adapt_transformers_to_gaudi()
    if SDP_ON_BF16 == 1:
        torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)
    if model_type == "gpt_bigcode":
        return StarCoder(model_id=model_id, revision=revision, dtype=dtype)
    if model_type == "bloom":
        return BLOOM(
            model_id=model_id,
            revision=revision,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "llava_next":
        return VlmCausalLM(
            model_class=LlavaNextForConditionalGeneration,
            model_id=model_id,
            revision=revision,
            quantize=None,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "mllama":
        return VlmCausalLM(
            model_class=MllamaForConditionalGeneration,
            model_id=model_id,
            revision=revision,
            quantize=None,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(
            model_id,
            revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    raise ValueError(f"Unsupported model type {model_type}")


# get_model_with_lora_adapters wraps the internal get_model function and adds support for loading adapters
# this provides a post model loading hook to load adapters into the model after the model has been loaded
def get_model_with_lora_adapters(
    model_id: str,
    lora_adapters: Optional[List[AdapterInfo]],
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    max_input_tokens: int,
    adapter_to_index: Dict[str, int],
):
    lora_adapter_ids = [adapter.id for adapter in lora_adapters]
    model = get_model(
        model_id,
        lora_adapter_ids,
        revision,
        sharded,
        quantize,
        speculate,
        dtype,
        trust_remote_code,
        max_input_tokens,
    )

    if len(lora_adapters) > 0:
        target_to_layer = build_layer_weight_lookup(model.model)

        for index, adapter in enumerate(lora_adapters):
            # The AdapterParameters object allows for merging multiple adapters into a single adapter.
            # At the moment, we only support loading a single adapter into the model, but we keep the
            # AdapterParameters object for easier extension in the future.
            adapter_parameters = AdapterParameters(
                adapter_info=[adapter],
                # when merging multiple adapters we can weight them differently
                # if this is not set, all adapters will be weighted equally
                # see: text_generation_server.utils.merges.strategies for impl
                weights=None,
                merge_strategy=0,
                density=1.0,
                majority_sign_method=0,
            )

            adapter_index = index + 1
            adapter_to_index[adapter.id] = adapter_index

            logger.info(
                f"Loading adapter weights into model: {','.join([adapter.id for adapter in adapter_parameters.adapter_info])}"
            )
            weight_names = tuple([v[0] for v in target_to_layer.values()])
            (
                module_map,
                adapter_config,
                adapter_weight_names,
                adapter_tokenizer,
            ) = load_and_merge_adapters(
                model.model_id,
                adapter_parameters,
                adapter_index,
                weight_names,
                False,
            )

            unused_weight_names = adapter_weight_names.copy()

            adapter_layers = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "qkv_proj",
            ]

            for layer_name in adapter_layers:
                nlayers = (
                    1 if layer_name == "lm_head" else len(model.model.model.layers)
                )
                adapter_weights = LoraWeights.prepare_weights(
                    config=adapter_config,
                    module_map=module_map,
                    layer_type=layer_name,
                    unused_weight_names=unused_weight_names,
                    nlayers=nlayers,
                    dtype=model.dtype,
                    world_size=model.world_size,
                    process_group=model.process_group,
                    target_to_layer=target_to_layer,
                )

                if adapter_weights is None:
                    continue

                model.layer_to_adapter_weights[layer_name].add_adapter(
                    adapter_index, adapter_weights
                )

            if len(unused_weight_names) > 0:
                logger.warning(
                    f"{','.join([a.id for a in lora_adapters])} unused adapter weights: {unused_weight_names}"
                )

            if adapter_tokenizer is not None:
                model.tokenizers.add_tokenizer(adapter_index, adapter_tokenizer)

            model.loaded_adapters.add(adapter_index)

    return model
