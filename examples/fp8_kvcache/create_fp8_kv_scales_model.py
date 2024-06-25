# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from examples/quantization/hf_ptq.py
"""

import argparse
import copy
import json
import random
import time
from safetensors.torch import safe_open

import ammo.torch.quantization as atq
import numpy as np
import torch
from ammo.torch.export import export_model_config
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import tempfile

RAND_SEED = 1234
MAX_SEQ_LEN = 2048

QUANT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"enable": False},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
        "*.query_key_value.output_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "*.Wqkv.output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*.W_pack.output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*.c_attn.output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*.k_proj.output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*.v_proj.output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
    },
    "algorithm": "max",
}


MODEL_NAME_PATTERN_MAP = {
    "Llama": "llama",
    "Mistral": "llama",
    "baichuan": "baichuan",
    "QWen": "qwen",
}


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, model_type=None):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )
    if model_type and model_type == "qwen":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16" or dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "fp16" or dtype == "float16":
        dtype = torch.float16
    elif dtype == "fp32" or dtype == "float32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    model_kwargs = {"torch_dtype": "auto"}

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, device_map="auto", **model_kwargs, trust_remote_code=True
    )
    model.eval()

    model_dtype = next(model.parameters()).dtype
    if dtype != model_dtype:
        print(
            "[TensorRT-LLM][WARNING] The manually set model data type is "
            f"{dtype}, but the data type of the HuggingFace model is "
            f"{model_dtype}."
        )

    return model


def get_model_type(model):
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_calib_dataloader(
    data="cnn_dailymail",
    tokenizer=None,
    batch_size=1,
    calib_size=512,
    block_size=512,
    device=None,
):
    print("Loading calibration dataset")
    if data == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train",
        )
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer.batch_encode_plus(
        dataset,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=block_size,
    )
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def quantize_model(model, quant_cfg, num_calib_samples, calib_dataloader=None):

    def calibrate_loop():
        if calib_dataloader is None:
            return
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in tqdm.tqdm(
            enumerate(calib_dataloader), total=num_calib_samples
        ):
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print("Quantization done. Total time used: {:.2f} s.".format(end_time - start_time))

    return model


def set_kv_scales(model, scales):
    for i, scale in scales.items():
        scale_param = torch.nn.Parameter(torch.tensor(scale), requires_grad=False)
        model.model.layers[int(i)].self_attn.kv_scale = scale_param

        if hasattr(model.model.layers[int(i)].self_attn.k_proj, "output_quantizer"):
            del model.model.layers[int(i)].self_attn.k_proj.output_quantizer
        if hasattr(model.model.layers[int(i)].self_attn.v_proj, "output_quantizer"):
            del model.model.layers[int(i)].self_attn.v_proj.output_quantizer


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    model = get_model(args.model_dir, args.dtype, args.device)
    model_type = get_model_type(model)
    tokenizer = get_tokenizer(args.model_dir, model_type=model_type)

    calib_dataloader = get_calib_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        calib_size=args.calib_size,
        device=args.device,
    )

    model = quantize_model(model, QUANT_CONFIG, args.calib_size, calib_dataloader)

    with torch.inference_mode():
        if model_type is None:
            print(
                f"Unknown model type {type(model).__name__}. Continue " "exporting..."
            )
            model_type = f"unknown:{type(model).__name__}"

        export_path = args.output_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            # export safetensors
            export_model_config(
                model,
                model_type,
                getattr(torch, args.dtype),
                export_dir=temp_dir,
                inference_tensor_parallel=1,
                inference_pipeline_parallel=1,
                export_tensorrt_llm_config=False,
                export_npz=False,
            )

            def load_safetensor(filename: str):
                with safe_open(filename, framework="pt") as f:
                    for name in f.keys():
                        param = f.get_tensor(name)
                        yield name, param

            layer_scales_map = {}
            for name, param in load_safetensor(temp_dir + "/rank0.safetensors"):
                if "kv_cache" in name:
                    nums = [int(s) for s in name.split(".") if s.isdecimal()]
                    if len(nums) != 1:
                        raise ValueError(f"Could not determine layer idx for {name}")

                    layer_idx = nums[0]
                    layer_scales_map[layer_idx] = param.item()

            set_kv_scales(model, layer_scales_map)
            model.config.kv_cache_dtype = "float8_e4m3fn"

            model.save_pretrained(export_path)
            tokenizer.save_pretrained(export_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir", help="Specify where the HuggingFace model is", required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", help="Model data type.", default="float16")
    parser.add_argument(
        "--batch_size", help="Batch size for calibration.", type=int, default=1
    )
    parser.add_argument(
        "--calib_size", help="Number of samples for calibration.", type=int, default=512
    )
    parser.add_argument("--output_dir", default="exported_model")
    args = parser.parse_args()

    main(args)
