import torch
import torch.distributed

from accelerate import init_empty_weights
from opentelemetry import trace
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from typing import Optional, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_santacoder_modeling import (
    FlashSantacoderForCausalLM
)
from text_generation_server.utils import (
    weight_files,
    download_weights,
    weight_hub_files,
    LocalEntryNotFoundError,
)

tracer = trace.get_tracer(__name__)


class FlashSantacoder(FlashCausalLM):
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashSantacoder is only available on GPU")

        if quantize:
            raise NotImplementedError("FlashSantacoder does not support quantization")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision,
            trust_remote_code=True  # Needed as the config is not part of Transformers
        )

        # We do not use from_pretrained as we modified the model internal module layout
        try:
            filenames = weight_files(model_id, revision, ".bin")
        # Local files not found
        except LocalEntryNotFoundError:
            hub_files = weight_hub_files(model_id, revision, ".bin")
            filenames = download_weights(hub_files, model_id, revision)

        with init_empty_weights():
            model = FlashSantacoderForCausalLM(config)

        self.load_weights(
            model,
            filenames,
        )
        self.model = model.eval().to(device).to(dtype)

        super(FlashCausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def load_weights(
            model: FlashSantacoderForCausalLM,
            filenames: List[Path],
    ):
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                layer_name = ".".join(key.split(".")[:4])

                # Fused qkv
                if "q_attn.weight" in key or "kv_attn.weight" in key:
                    final_key = layer_name + ".attn.weight"
                elif "q_attn.bias" in key or "kv_attn.bias" in key:
                    final_key = layer_name + ".attn.bias"

                else:
                    final_key = key

                module_name, param_name = final_key.rsplit(".", 1)
                module = model.get_submodule(module_name)

                try:
                    current_parameter_tensor = module._parameters[param_name]
                except KeyError:
                    current_parameter_tensor = None

                if current_parameter_tensor is not None:
                    if "c_fc.weight" in key or "c_proj.weight" in key or "q_attn.weight" in key or "kv_attn.weight" in key:
                        # Tranpose as we use nn.Linear instead of Conv1D
                        value = value.T

                    if current_parameter_tensor.device == torch.device("meta"):
                        # Init qkv
                        if "attn.weight" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (model.transformer.head_size * (model.transformer.num_heads + 2), value.shape[1])
                            )
                        elif "attn.bias" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (model.transformer.head_size * (model.transformer.num_heads + 2))
                            )

                    # Copy to correct slice
                    if "q_attn.weight" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "q_attn.bias" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "kv_attn.weight" in key:
                        module._parameters[param_name][
                        model.transformer.head_size * model.transformer.num_heads:
                        ] = value
                    elif "kv_attn.bias" in key:
                        module._parameters[param_name][
                        model.transformer.head_size * model.transformer.num_heads:
                        ] = value
                    else:
                        if current_parameter_tensor.shape != value.shape:
                            raise ValueError(
                                f"Name {final_key} -- Current {current_parameter_tensor.shape} and got {value.shape}"
                            )
                        module._parameters[param_name] = value
                else:
                    module._buffers[param_name] = value

        torch.cuda.empty_cache()
        model.post_load_weights()

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, cleanup_tokenization_spaces=False
        )
