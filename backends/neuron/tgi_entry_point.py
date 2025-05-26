#!/usr/bin/env python

import logging
import os
import sys


from text_generation_server.tgi_env import (
    available_cores,
    get_env_dict,
    get_neuron_config_for_model,
    neuron_config_to_env,
    neuronxcc_version,
    parse_cmdline_and_set_env,
    tgi_env_vars,
)


logger = logging.getLogger(__name__)


def main():
    """
    This script determines proper default TGI env variables for the neuron precompiled models to
    work properly
    :return:
    """
    args = parse_cmdline_and_set_env()

    for env_var in tgi_env_vars:
        if not os.getenv(env_var):
            break
    else:
        logger.info(
            "All env vars %s already set, skipping, user know what they are doing",
            tgi_env_vars,
        )
        sys.exit(0)

    neuron_config = get_neuron_config_for_model(args.model_id, args.revision)

    if not neuron_config:
        msg = (
            "No compatible neuron config found. Provided env {}, available cores {}, neuronxcc version {}"
        ).format(get_env_dict(), available_cores, neuronxcc_version)
        logger.error(msg)
        raise Exception(msg)

    neuron_config_to_env(neuron_config)


if __name__ == "__main__":
    main()
