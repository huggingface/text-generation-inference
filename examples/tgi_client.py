# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import os
import statistics
import threading
import time
import tqdm
from typing import List

from huggingface_hub import InferenceClient


def except_hook(args):
    print(f"Thread failed with error: {args.exc_value}")
    os._exit(1)

threading.excepthook = except_hook


class TgiClient:
    def __init__(
        self,
        server_address: str,
        max_num_threads: int
    ) -> None:
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_num_threads)
        self._client = InferenceClient(server_address)

        self._ttft = []
        self._tpot = []
        self._generated_tokens = []

    def run_generation(
        self,
        samples: List[str],
        max_new_tokens: int
    ) ->  None:
        """
        Run generation for every sample in dataset.
        Creates a separate thread for every sample.
        """
        threads: List[threading.Thread] = []
        for sample in tqdm.tqdm(samples):
            self._semaphore.acquire()
            threads.append(
                threading.Thread(
                    target=self._process_sample, args=[sample, max_new_tokens]
                )
            )
            threads[-1].start()
        for thread in threads:
            if thread is not None:
                thread.join()

    def _process_sample(
        self,
        sample: str,
        max_new_tokens: int
    ) -> None:
        """
        Generates response stream for a single sample.
        Collects performance metrics.
        """
        timestamp = time.perf_counter_ns()
        response_stream = self._client.text_generation(
            sample, max_new_tokens=max_new_tokens, stream=True, details=True
        )
        out = ''
        for id, response in enumerate(response_stream):
            if id == 0:
                self._ttft.append(time.perf_counter_ns() - timestamp)
            else:
                self._tpot.append(time.perf_counter_ns() - timestamp)
            timestamp = time.perf_counter_ns()
            out += response.token.text
            if response.details:
                self._generated_tokens.append(response.details.generated_tokens)

        self._semaphore.release()

    def print_performance_metrics(
        self,
        duration_s: float
    ) -> None:
        def line():
            print(32*"-")

        line()
        print("----- Performance  summary -----")
        line()
        print(f"Throughput: {sum(self._generated_tokens) / duration_s:.1f} tokens/s")
        print(f"Throughput: {len(self._generated_tokens) / duration_s:.1f} queries/s")
        line()
        print(f"First token latency:")
        print(f"\tMedian: \t{statistics.median(self._ttft)*1e-6:.2f}ms")
        print(f"\tAverage: \t{statistics.fmean(self._ttft)*1e-6:.2f}ms")
        line()
        print(f"Output token latency:")
        print(f"\tMedian: \t{statistics.median(self._tpot)*1e-6:.2f}ms")
        print(f"\tAverage: \t{statistics.fmean(self._tpot)*1e-6:.2f}ms")
        line()
