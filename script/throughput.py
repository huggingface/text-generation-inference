from concurrent import futures
from typing import Any, Dict, List
import time

import requests
import json


# copied from other benchmarking script
def simple_throughput(
    predictor: Any, payloads: List[Dict[str, Any]], concurrent_requests: int
):
    time_start = time.time()
    with futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        responses = list(executor.map(predictor.predict, payloads))

    if isinstance(responses[0], list):
        responses = [response[0] for response in responses]

    total_tokens = sum(
        [
            len([token for token in x["details"]["tokens"] if token["id"] != -1])
            for x in responses
        ]
    )
    token_throughput = total_tokens / (time.time() - time_start)
    generated_tokens_per_request = total_tokens / len(payloads)
    print(f"concurrent requests: {concurrent_requests}")
    print(f"throughput: {token_throughput:.2f}")
    print(
        f"generated tokens per request: {generated_tokens_per_request:.2f}", end="\n\n"
    )


# setup to help run
real = True


class Predictor:
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response_json = {"details": {"tokens": []}}
        if real:
            url = "http://localhost:3000/generate"
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            response_json = response.json()
        else:
            time.sleep(0.1)

        return response_json


max_new_tokens = 100

if __name__ == "__main__":
    print("Running throughput test")
    predictor = Predictor()
    payloads = [
        {
            "inputs": "<s>[INST] I am making mayonnaise, it was starting to thicken but now it has become runny and liquid again, is there any way to salvage it? [/INST]Yes, it's possible to fix runny mayonnaise! The most common reason for mayonnaise becoming runny is because the oil was added too quickly or the egg yolk wasn't emulsified properly. Here are some steps you can take to fix it:\n\n1. Separate another egg yolk and place it in a clean, dry bowl.\n2. Slowly add the runny mayonnaise to the egg yolk while whisking vigorously.\n3. Once all the runny mayonnaise has been added, continue whisking until the mixture has emulsified and thickened.\n4. If the mayonnaise is still too runny, you can add another egg yolk and repeat the process.\n\nIf the mayonnaise still won't thicken, you can try adding a small amount of dijon mustard or vinegar to the mixture, which can act as emulsifiers and help stabilize the mayonnaise. It's important to add these ingredients slowly and in small amounts to avoid over-thinning the mixture.</s>[INST] What is optimal Mayonnaise thickness? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] Why Aristotelian view of physics (impetus and stuff) is wrong? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] If you were to image the experience of eating from only others descriptions and never having done it yourself, how would you describe it. [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] What is the best way to cook a steak? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] How do you make a perfect omelette? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] What is the secret to a good pizza dough? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] How do you make a classic French onion soup? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] What is the best way to roast a chicken? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] How do you make a perfect chocolate cake? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] What is the secret to a good risotto? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
        {
            "inputs": "<s>[INST] How do you make a classic spaghetti carbonara? [/INST]",
            "parameters": {"details": True, "max_new_tokens": max_new_tokens},
        },
    ]

    for concurrent_requests in [1, 2, 4, 8]:
        simple_throughput(predictor, payloads, concurrent_requests)
    print("Throughput test complete")
