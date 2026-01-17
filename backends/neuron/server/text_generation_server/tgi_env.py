#!/usr/bin/env python

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from optimum.neuron.modeling_decoder import get_available_cores
from optimum.neuron.cache import get_hub_cached_entries
from optimum.neuron.configuration_utils import NeuronConfig
from optimum.neuron.utils.version_utils import get_neuronxcc_version
from optimum.neuron.utils import map_torch_dtype


logger = logging.getLogger(__name__)

tgi_router_env_vars = [
    "MAX_BATCH_SIZE",
    "MAX_TOTAL_TOKENS",
    "MAX_INPUT_TOKENS",
    "MAX_BATCH_PREFILL_TOKENS",
]
tgi_server_env_vars = ["HF_NUM_CORES", "HF_AUTO_CAST_TYPE"]


# By the end of this script all env var should be specified properly
tgi_env_vars = tgi_server_env_vars + tgi_router_env_vars

available_cores = get_available_cores()
neuronxcc_version = get_neuronxcc_version()


def parse_cmdline_and_set_env(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    if not argv:
        argv = sys.argv
    # All these are params passed to tgi and intercepted here
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=os.getenv("MAX_INPUT_TOKENS", os.getenv("MAX_INPUT_LENGTH", 0)),
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=os.getenv("MAX_TOTAL_TOKENS", 0)
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=os.getenv("MAX_BATCH_SIZE", 0)
    )
    parser.add_argument(
        "--max-batch-prefill-tokens",
        type=int,
        default=os.getenv("MAX_BATCH_PREFILL_TOKENS", 0),
    )
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID"))
    parser.add_argument("--revision", type=str, default=os.getenv("REVISION"))

    args = parser.parse_known_args(argv)[0]

    if not args.model_id:
        raise Exception(
            "No model id provided ! Either specify it using --model-id cmdline or MODEL_ID env var"
        )

    # Override env with cmdline params
    os.environ["MODEL_ID"] = args.model_id

    # Set all tgi router and tgi server values to consistent values as early as possible
    # from the order of the parser defaults, the tgi router value can override the tgi server ones
    if args.max_total_tokens > 0:
        os.environ["MAX_TOTAL_TOKENS"] = str(args.max_total_tokens)

    if args.max_input_tokens > 0:
        os.environ["MAX_INPUT_TOKENS"] = str(args.max_input_tokens)

    if args.max_batch_size > 0:
        os.environ["MAX_BATCH_SIZE"] = str(args.max_batch_size)

    if args.max_batch_prefill_tokens > 0:
        os.environ["MAX_BATCH_PREFILL_TOKENS"] = str(args.max_batch_prefill_tokens)

    if args.revision:
        os.environ["REVISION"] = str(args.revision)

    return args


def neuron_config_to_env(neuron_config):
    if isinstance(neuron_config, NeuronConfig):
        neuron_config = neuron_config.to_dict()
    with open(os.environ["ENV_FILEPATH"], "w") as f:
        f.write("export MAX_BATCH_SIZE={}\n".format(neuron_config["batch_size"]))
        f.write("export MAX_TOTAL_TOKENS={}\n".format(neuron_config["sequence_length"]))
        f.write("export HF_NUM_CORES={}\n".format(neuron_config["tp_degree"]))
        config_key = (
            "auto_cast_type" if "auto_cast_type" in neuron_config else "torch_dtype"
        )
        auto_cast_type = neuron_config[config_key]
        f.write("export HF_AUTO_CAST_TYPE={}\n".format(auto_cast_type))
        max_input_tokens = os.getenv("MAX_INPUT_TOKENS")
        if not max_input_tokens:
            max_input_tokens = int(neuron_config["sequence_length"]) // 2
            if max_input_tokens == 0:
                raise Exception("Model sequence length should be greater than 1")
        f.write("export MAX_INPUT_TOKENS={}\n".format(max_input_tokens))
        max_batch_prefill_tokens = os.getenv("MAX_BATCH_PREFILL_TOKENS")
        if not max_batch_prefill_tokens:
            max_batch_prefill_tokens = int(neuron_config["batch_size"]) * int(
                max_input_tokens
            )
        f.write("export MAX_BATCH_PREFILL_TOKENS={}\n".format(max_batch_prefill_tokens))


def sort_neuron_configs(dictionary):
    return -dictionary["tp_degree"], dictionary["batch_size"]


def lookup_compatible_cached_model(
    model_id: str, revision: Optional[str]
) -> Optional[Dict[str, Any]]:
    # Reuse the same mechanic as the one in use to configure the tgi server part
    # The only difference here is that we stay as flexible as possible on the compatibility part
    entries = get_hub_cached_entries(model_id)

    logger.debug(
        "Found %d cached entries for model %s, revision %s",
        len(entries),
        model_id,
        revision,
    )

    all_compatible = []
    for entry in entries:
        if check_env_and_neuron_config_compatibility(
            entry, check_compiler_version=True
        ):
            all_compatible.append(entry)

    if not all_compatible:
        logger.debug(
            "No compatible cached entry found for model %s, env %s, available cores %s, neuronxcc version %s",
            model_id,
            get_env_dict(),
            available_cores,
            neuronxcc_version,
        )
        return None

    logger.info("%d compatible neuron cached models found", len(all_compatible))

    all_compatible = sorted(all_compatible, key=sort_neuron_configs)

    entry = all_compatible[0]

    return entry


def check_env_and_neuron_config_compatibility(
    neuron_config_dict: Dict[str, Any], check_compiler_version: bool
) -> bool:
    logger.debug(
        "Checking the provided neuron config %s is compatible with the local setup and provided environment",
        neuron_config_dict,
    )

    # Local setup compat checks
    if neuron_config_dict["tp_degree"] > available_cores:
        logger.debug(
            "Not enough neuron cores available to run the provided neuron config"
        )
        return False

    if (
        check_compiler_version
        and neuron_config_dict["neuronxcc_version"] != neuronxcc_version
    ):
        logger.debug(
            "Compiler version conflict, the local one (%s) differs from the one used to compile the model (%s)",
            neuronxcc_version,
            neuron_config_dict["neuronxcc_version"],
        )
        return False

    batch_size = os.getenv("MAX_BATCH_SIZE", None)
    if batch_size is not None and neuron_config_dict["batch_size"] < int(batch_size):
        logger.debug(
            "The provided MAX_BATCH_SIZE (%s) is higher than the neuron config batch size (%s)",
            os.getenv("MAX_BATCH_SIZE"),
            neuron_config_dict["batch_size"],
        )
        return False
    max_total_tokens = os.getenv("MAX_TOTAL_TOKENS", None)
    if max_total_tokens is not None and neuron_config_dict["sequence_length"] < int(
        max_total_tokens
    ):
        logger.debug(
            "The provided MAX_TOTAL_TOKENS (%s) is higher than the neuron config sequence length (%s)",
            max_total_tokens,
            neuron_config_dict["sequence_length"],
        )
        return False
    num_cores = os.getenv("HF_NUM_CORES", None)
    if num_cores is not None and neuron_config_dict["tp_degree"] < int(num_cores):
        logger.debug(
            "The provided HF_NUM_CORES (%s) is higher than the neuron config tp degree (%s)",
            num_cores,
            neuron_config_dict["tp_degree"],
        )
        return False
    auto_cast_type = os.getenv("HF_AUTO_CAST_TYPE", None)
    if auto_cast_type is not None:
        config_key = (
            "auto_cast_type"
            if "auto_cast_type" in neuron_config_dict
            else "torch_dtype"
        )
        neuron_config_value = map_torch_dtype(str(neuron_config_dict[config_key]))
        env_value = map_torch_dtype(auto_cast_type)
        if env_value != neuron_config_value:
            logger.debug(
                "The provided auto cast type and the neuron config param differ (%s != %s)",
                env_value,
                neuron_config_value,
            )
            return False
    max_input_tokens = int(
        os.getenv("MAX_INPUT_TOKENS", os.getenv("MAX_INPUT_LENGTH", 0))
    )
    if max_input_tokens > 0:
        if hasattr(neuron_config_dict, "max_context_length"):
            sequence_length = neuron_config_dict["max_context_length"]
        else:
            sequence_length = neuron_config_dict["sequence_length"]
        if max_input_tokens >= sequence_length:
            logger.debug(
                "Specified max input tokens is not compatible with config sequence length ( %s >= %s)",
                max_input_tokens,
                sequence_length,
            )
            return False

    return True


def get_env_dict() -> Dict[str, str]:
    d = {}
    for k in tgi_env_vars:
        d[k] = os.getenv(k)
    return d


def get_neuron_config_for_model(
    model_name_or_path: str, revision: Optional[str] = None
) -> NeuronConfig:
    try:
        neuron_config = NeuronConfig.from_pretrained(
            model_name_or_path, revision=revision
        )
    except Exception as e:
        logger.debug(
            "NeuronConfig.from_pretrained failed for model %s, revision %s: %s",
            model_name_or_path,
            revision,
            e,
        )
        neuron_config = None
    if neuron_config is not None:
        compatible = check_env_and_neuron_config_compatibility(
            neuron_config.to_dict(), check_compiler_version=False
        )
        if not compatible:
            env_dict = get_env_dict()
            msg = (
                "Invalid neuron config and env. Config {}, env {}, available cores {}, neuronxcc version {}"
            ).format(neuron_config, env_dict, available_cores, neuronxcc_version)
            logger.error(msg)
            raise Exception(msg)
    else:
        neuron_config = lookup_compatible_cached_model(model_name_or_path, revision)

    return neuron_config
