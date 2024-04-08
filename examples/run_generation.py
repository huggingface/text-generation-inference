# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import argparse
import requests
import time
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer

from tgi_client import TgiClient


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_address", type=str, default="http://localhost:8080", help="Address of the TGI server"
    )
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model id used in TGI server"
    )
    parser.add_argument(
        "--max_input_length", type=int, default=1024, help="Max input length for TGI model"
    )
    parser.add_argument(
        "--max_output_length", type=int, default=1024, help="Max output length for TGI model"
    )
    parser.add_argument(
        "--total_sample_count", type=int, default=2048, help="Total number of samples to generate"
    )
    parser.add_argument(
        "--max_concurrent_requests", type=int, default=256, help="Max number of concurrent requests"
    )
    return parser.parse_args()


def read_dataset(
    max_input_length: int,
    total_sample_count: int,
    model_id: str
) -> List[str]:
    """
    Loads public dataset from HF: https://huggingface.co/datasets/DIBT/10k_prompts_ranked
    and filters out too long samples.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = load_dataset("DIBT/10k_prompts_ranked", split="train", trust_remote_code=True)
    dataset = dataset.filter(
        lambda x: len(tokenizer(x["prompt"])["input_ids"]) < max_input_length
    )
    if len(dataset) > total_sample_count:
        dataset = dataset.select(range(total_sample_count))
    dataset = dataset.shuffle()
    return [sample["prompt"] for sample in dataset]


def is_tgi_available(
    server_address: str
) -> bool:
    """
    Checks if TGI server is available under the specified address.
    """
    try:
        info = requests.get(f"{server_address}/info")
        return info.status_code == 200
    except:
        return False


def main():
    args = get_args()
    dataset = read_dataset(
        args.max_input_length, args.total_sample_count, args.model_id
    )

    if not is_tgi_available(args.server_address):
        raise RuntimeError("Cannot connect with TGI server!")

    tgi_client = TgiClient(
        args.server_address, args.max_concurrent_requests
    )
    timestamp = time.perf_counter_ns()
    tgi_client.run_generation(
        dataset, args.max_output_length
    )
    duration_s = (time.perf_counter_ns() - timestamp) * 1e-9
    tgi_client.print_performance_metrics(duration_s)


if __name__ == '__main__':
    main()
