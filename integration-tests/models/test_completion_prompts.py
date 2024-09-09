import pytest
import requests
import json
from aiohttp import ClientSession

from text_generation.types import (
    Completion,
)


@pytest.fixture(scope="module")
def flash_llama_completion_handle(launcher):
    with launcher(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_completion(flash_llama_completion_handle):
    await flash_llama_completion_handle.health(300)
    return flash_llama_completion_handle.client


# NOTE: since `v1/completions` is a deprecated inferface/endpoint we do not provide a convience
# method for it. Instead, we use the `requests` library to make the HTTP request directly.


@pytest.mark.release
def test_flash_llama_completion_single_prompt(
    flash_llama_completion, response_snapshot
):
    response = requests.post(
        f"{flash_llama_completion.base_url}/v1/completions",
        json={
            "model": "tgi",
            "prompt": "What is Deep Learning?",
            "max_tokens": 5,
            "seed": 0,
        },
        headers=flash_llama_completion.headers,
        stream=False,
    )
    response = response.json()
    assert len(response["choices"]) == 1
    assert response["choices"][0]["text"] == "\n2.2 How"
    assert response == response_snapshot


@pytest.mark.release
def test_flash_llama_completion_many_prompts(flash_llama_completion, response_snapshot):
    response = requests.post(
        f"{flash_llama_completion.base_url}/v1/completions",
        json={
            "model": "tgi",
            "prompt": ["Say", "this", "is", "a"],
            "max_tokens": 10,
            "seed": 0,
        },
        headers=flash_llama_completion.headers,
        stream=False,
    )
    response = response.json()
    assert len(response["choices"]) == 4

    all_indexes = [choice["index"] for choice in response["choices"]]
    all_indexes.sort()
    assert all_indexes == [0, 1, 2, 3]

    assert response == response_snapshot


@pytest.mark.release
async def test_flash_llama_completion_many_prompts_stream(
    flash_llama_completion, response_snapshot
):
    request = {
        "model": "tgi",
        "prompt": [
            "What is Deep Learning?",
            "Is water wet?",
            "What is the capital of France?",
            "def mai",
        ],
        "max_tokens": 10,
        "seed": 0,
        "stream": True,
    }

    url = f"{flash_llama_completion.base_url}/v1/completions"

    chunks = []
    strings = [""] * 4
    async with ClientSession(headers=flash_llama_completion.headers) as session:
        async with session.post(url, json=request) as response:
            # iterate over the stream
            async for chunk in response.content.iter_any():
                # remove "data:"
                chunk = chunk.decode().split("\n\n")
                # remove "data:" if present
                chunk = [c.replace("data:", "") for c in chunk]
                # remove empty strings
                chunk = [c for c in chunk if c]
                # remove completion marking chunk
                chunk = [c for c in chunk if c != " [DONE]"]
                # parse json
                chunk = [json.loads(c) for c in chunk]

                for c in chunk:
                    chunks.append(Completion(**c))
                    assert "choices" in c
                    index = c["choices"][0]["index"]
                    assert 0 <= index <= 4
                    strings[index] += c["choices"][0]["text"]

    assert response.status == 200
    # assert strings == ["What Business: And Stock Mohs`('\\", '\nrig Business Process And Stock ,s, And', '\n\n202 Stock Mohs a Service', 'hd\n20207\nR1']
    assert chunks == response_snapshot
