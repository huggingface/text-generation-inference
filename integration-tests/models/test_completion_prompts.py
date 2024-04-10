import pytest
import requests
import json
from aiohttp import ClientSession


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


def test_flash_llama_completion_single_prompt(
    flash_llama_completion, response_snapshot
):
    response = requests.post(
        f"{flash_llama_completion.base_url}/v1/completions",
        json={
            "model": "tgi",
            "prompt": "Say this is a test",
            "max_tokens": 5,
            "seed": 0,
        },
        headers=flash_llama_completion.headers,
        stream=False,
    )
    response = response.json()
    assert len(response["choices"]) == 1


def test_flash_llama_completion_many_prompts(flash_llama_completion, response_snapshot):
    response = requests.post(
        f"{flash_llama_completion.base_url}/v1/completions",
        json={
            "model": "tgi",
            "prompt": ["Say", "this", "is", "a", "test"],
            "max_tokens": 5,
            "seed": 0,
        },
        headers=flash_llama_completion.headers,
        stream=False,
    )
    response = response.json()
    assert len(response["choices"]) == 5

    all_indexes = [choice["index"] for choice in response["choices"]]
    all_indexes.sort()
    assert all_indexes == [0, 1, 2, 3, 4]


async def test_flash_llama_completion_many_prompts_stream(
    flash_llama_completion, response_snapshot
):
    request = {
        "model": "tgi",
        "prompt": ["Say", "this", "is", "a", "test"],
        "max_tokens": 5,
        "seed": 0,
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
    }

    url = f"{flash_llama_completion.base_url}/v1/completions"

    async with ClientSession(headers=headers) as session:
        async with session.post(url, json=request) as resp:
            # iterate over the stream
            async for chunk in resp.content.iter_any():
                # strip data: prefix and convert to json
                data = json.loads(chunk.decode("utf-8")[5:])
                assert "choices" in data
                assert len(data["choices"]) == 1
                assert data["choices"][0]["index"] in [*range(len(request["prompt"]))]

    assert resp.status == 200
