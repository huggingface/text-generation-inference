def load_text_model(prefix, config, weights, name=None):
    if config.model_type == "llama":
        from text_generation_server.models.custom_modeling.flash_llama_modeling import (
            FlashLlamaForCausalLM,
        )

        return FlashLlamaForCausalLM(prefix, config, weights)
    elif config.model_type == "mistral":
        from text_generation_server.models.custom_modeling.flash_mistral_modeling import (
            FlashMistralForCausalLM,
        )

        return FlashMistralForCausalLM(prefix, config, weights, name=name)
    else:
        raise RuntimeError(f"Unsupported model type {config.model_type}")


def load_vision_model(prefix, config, weights):
    if config.model_type == "clip_vision_model":
        from text_generation_server.models.custom_modeling.clip import (
            CLIPVisionTransformer,
        )

        return CLIPVisionTransformer(
            prefix=f"{prefix}.vision_model", config=config, weights=weights
        )
    else:
        raise RuntimeError(f"Unsupported model type {config.model_type}")
