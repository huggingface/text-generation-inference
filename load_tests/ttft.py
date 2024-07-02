import asyncio
import aiohttp
import json
import os
from time import time

HOST = os.getenv("HOST", "localhost:3000")
MODEL_ID = os.getenv("MODEL_ID", "default-model")
NUM_REQUESTS = 10
MAX_NEW_TOKENS = 100
TIMEOUT = 30


def load_inputs(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    inputs = []
    for item in data:
        if "conversations" in item:
            if len(item["conversations"]) > 0:
                inputs.append(item["conversations"][0]["value"])

    return inputs


def generate_payload(input_text):
    return {
        "messages": [{"role": "user", "content": input_text}],
        "temperature": 0,
        "model": MODEL_ID,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": True,
    }


async def benchmark_sse(session, input_text):
    payload = generate_payload(input_text)
    start_time = time()
    first_token_time = None

    try:
        async with session.post(
            f"http://{HOST}/v1/chat/completions", json=payload, timeout=TIMEOUT
        ) as response:
            async for line in response.content:
                if line.startswith(b"data:"):
                    if first_token_time is None:
                        first_token_time = time()
                        return (first_token_time - start_time) * 1000

            if first_token_time is None:
                raise Exception("No SSE data received within the timeout period")

    except asyncio.TimeoutError:
        raise Exception(f"Request timed out after {TIMEOUT} seconds")


async def run_benchmark(inputs, same_input=False):
    async with aiohttp.ClientSession() as session:
        tasks = []
        longest_input = 0
        for i in range(NUM_REQUESTS):
            input_text = inputs[0] if same_input else inputs[i % len(inputs)]
            if len(input_text) > longest_input:
                longest_input = len(input_text)
            task = asyncio.create_task(benchmark_sse(session, input_text))
            tasks.append(task)

        results = []
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                time_to_first_event = await task
                results.append(time_to_first_event)
                print(
                    f"Request {i}: Time to first event - {time_to_first_event:.2f}ms longest input: {longest_input}"
                )
            except Exception as e:
                print(f"Request {i} failed: {str(e)}")

    if results:
        avg_time = sum(results) / len(results)
        print(f"\nAverage time to first event: {avg_time:.2f}ms")
    else:
        print("\nNo successful requests")

    return avg_time if results else None


async def main():
    inputs = load_inputs("small.json")

    print("Running benchmark with same input:")
    same_input_avg = await run_benchmark(inputs, same_input=True)

    # sleep for a second to avoid the next inputs in the same batch
    await asyncio.sleep(1)

    print("\nRunning benchmark with different inputs:")
    different_inputs_avg = await run_benchmark(inputs, same_input=False)

    if same_input_avg and different_inputs_avg:
        print(f"\nSame input average: {same_input_avg:.2f}ms")
        print(f"Different inputs average: {different_inputs_avg:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
