import os
import sys
import json
from loguru import logger

from transformers.models.auto import modeling_auto
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel

def download_and_unload_peft(model_id, revision, adapter_config_filename, trust_remote_code):

    logger.info("Peft model detected.")
    peft_model_id = model_id
    with open(adapter_config_filename, "r") as f:
        adapter_config = json.load(f)
    model_id = adapter_config["base_model_name_or_path"]
    logger.info(f"Merging the lora weights {repr(peft_model_id)} into the base model {repr(model_id)}")
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    model_type = config.model_type
    logger.info(f"Starting to load the base model {repr(model_id)}, this may take a while with no feedback.")
    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code
        )
    elif model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
        base_model = AutoModelForSeq2SeqLM(
            model_id,
            revision=revision,
        )
    else:
        # We have no idea just try either
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code
            )
        except Exception:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code
            )
    logger.info(f"Loaded.")

    logger.info(f"Merging the lora weights {repr(peft_model_id)} into the base model {repr(model_id)}")

    model = PeftModel.from_pretrained(
        base_model,
        peft_model_id,
        trust_remote_code=trust_remote_code,
    )
    model = model.merge_and_unload()
    
    os.makedirs(peft_model_id, exist_ok=True)
    cache_dir = peft_model_id
    logger.info(f"Saving the newly created merged model to {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(cache_dir, safe_serialization=True)
    config.save_pretrained(cache_dir)
    tokenizer.save_pretrained(cache_dir)



