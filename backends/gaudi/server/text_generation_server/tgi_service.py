import os
from pathlib import Path
from loguru import logger
import sys
from text_generation_server import server
import argparse
from typing import List
from text_generation_server.utils.adapter import parse_lora_adapters


def main(args):
    logger.info("TGIService: starting tgi service .... ")
    logger.info(
        "TGIService: --model_id {}, --revision {}, --sharded {}, --speculate {}, --dtype {}, --trust_remote_code {}, --uds_path {} ".format(
            args.model_id, args.revision, args.sharded, args.speculate, args.dtype, args.trust_remote_code, args.uds_path
        )
    )
    lora_adapters = parse_lora_adapters(os.getenv("LORA_ADAPTERS"))
    server.serve(
        model_id=args.model_id,
        lora_adapters=lora_adapters,
        revision=args.revision,
        sharded=args.sharded,
        quantize=args.quantize,
        speculate=args.speculate,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        uds_path=args.uds_path,
        max_input_tokens=args.max_input_tokens
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--revision", type=str)
    parser.add_argument("--sharded", type=bool)
    parser.add_argument("--speculate", type=int, default=None)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--trust_remote_code", type=bool)
    parser.add_argument("--uds_path", type=Path)
    parser.add_argument("--quantize", type=str)
    parser.add_argument("--max_input_tokens", type=int)
    args = parser.parse_args()
    main(args)
