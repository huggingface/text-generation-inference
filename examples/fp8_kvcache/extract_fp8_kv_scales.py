import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from text_generation_server.utils.hub import (
    weight_files,
    download_weights,
    weight_hub_files,
)
from safetensors import safe_open
import argparse


def load_model(ckpt_path):
    model_args = {"torch_dtype": "auto"}

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, device_map="auto", **model_args, trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    return model, tokenizer


def set_nested_attribute(obj, attribute_path, value):
    keys = attribute_path.split(".")
    current_obj = obj
    for key in keys[:-1]:
        current_obj = getattr(current_obj, key)
    setattr(current_obj, keys[-1], value)


def apply_kv_scales_to_model(model, layer_scales_map):
    for layer_name, scale_value in layer_scales_map.items():
        scale_param = torch.nn.Parameter(torch.tensor(scale_value), requires_grad=False)
        set_nested_attribute(model, layer_name, scale_param)


def extract_kv_scales(quantized_model):
    def fetch_parameters(filename):
        with safe_open(filename, framework="pt") as f:
            for name in f.keys():
                param_tensor = f.get_tensor(name)
                yield name, param_tensor

    checkpoint_dir = Path(quantized_model)
    if not checkpoint_dir.is_dir():
        hub_filenames = weight_hub_files(quantized_model)
        downloaded_files = download_weights(hub_filenames, quantized_model)
    downloaded_files = weight_files(quantized_model, extension=".safetensors")

    layer_scales_map = {}
    for tensor_file in downloaded_files:
        for name, param in fetch_parameters(tensor_file):
            if ".kv_scale" in name:
                layer_scales_map[name] = param.item()

    return layer_scales_map


def main(quantized_model, model_id, save_path):
    layer_scales_map = extract_kv_scales(quantized_model)

    model, tokenizer = load_model(model_id)

    apply_kv_scales_to_model(model, layer_scales_map)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract FP8 KV cache scales and add them to a FP16 model."
    )
    parser.add_argument(
        "--quantized-model",
        type=str,
        help="Path to the FP8 model checkpoint to extract KV cache scales",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID of the FP16 model to save the KV cache scales",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the FP16 model with the kv scales",
    )

    args = parser.parse_args()

    main(args.quantized_model, args.model, args.save_path)
