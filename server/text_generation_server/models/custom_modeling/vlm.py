def load_text_model(prefix, config, weights, name=None):
    if config.model_type == "llama":
        from text_generation_server.models.custom_modeling.flash_llama_modeling import (
            FlashLlamaForCausalLM,
        )

        return FlashLlamaForCausalLM(prefix, config, weights, name=name)
    elif config.model_type == "mistral":
        from text_generation_server.models.custom_modeling.flash_mistral_modeling import (
            FlashMistralForCausalLM,
        )

        return FlashMistralForCausalLM(prefix, config, weights, name=name)
    elif config.model_type == "gemma":
        from text_generation_server.models.custom_modeling.flash_gemma_modeling import (
            FlashGemmaForCausalLM,
        )

        return FlashGemmaForCausalLM(prefix, config, weights, causal=False)
    elif config.model_type == "gemma2":
        from text_generation_server.models.custom_modeling.flash_gemma2_modeling import (
            FlashGemma2ForCausalLM,
        )

        return FlashGemma2ForCausalLM(prefix, config, weights)

    elif config.model_type == "gemma3" or config.model_type == "gemma3_text":
        from text_generation_server.models.custom_modeling.flash_gemma3_modeling import (
            FlashGemma3ForCausalLM,
        )

        return FlashGemma3ForCausalLM(prefix, config, weights)
    elif config.model_type == "paligemma":
        from text_generation_server.models.custom_modeling.flash_gemma_modeling import (
            FlashGemmaForCausalLM,
        )

        return FlashGemmaForCausalLM(prefix, config, weights)
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
    if (
        config.model_type == "siglip_vision_model"
        or config.model_type == "gemma3_vision"
    ):
        from text_generation_server.models.custom_modeling.siglip import (
            SiglipVisionTransformer,
        )

        # TODO: ensure that using the prefix doesn't break any existing models
        # that rely on the old prefix (update the old models if necessary)
        return SiglipVisionTransformer(
            # prefix="vision_model.vision_model", config=config, weights=weights
            prefix=f"{prefix}.vision_model",
            config=config,
            weights=weights,
        )
    else:
        raise RuntimeError(f"Unsupported model type {config.model_type}")
