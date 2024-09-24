import pytest
import requests
import json
from aiohttp import ClientSession

from text_generation.types import Completion, ChatCompletionChunk


@pytest.fixture(scope="module")
def flash_llama_completion_handle(launcher):
    with launcher(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
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
            "max_tokens": 10,
            "temperature": 0.0,
        },
        headers=flash_llama_completion.headers,
        stream=False,
    )
    response = response.json()
    assert len(response["choices"]) == 1
    assert (
        response["choices"][0]["text"]
        == " A Beginner’s Guide\nDeep learning is a subset"
    )
    assert response == response_snapshot


@pytest.mark.release
async def test_flash_llama_completion_stream_usage(
    flash_llama_completion, response_snapshot
):
    url = f"{flash_llama_completion.base_url}/v1/chat/completions"
    request = {
        "model": "tgi",
        "messages": [
            {
                "role": "user",
                "content": "What is Deep Learning?",
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream_options": {"include_usage": True},
        "stream": True,
    }
    string = ""
    chunks = []
    had_usage = False
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
                    chunks.append(ChatCompletionChunk(**c))
                    assert "choices" in c
                    if len(c["choices"]) == 1:
                        index = c["choices"][0]["index"]
                        assert index == 0
                        string += c["choices"][0]["delta"]["content"]

                        has_usage = c["usage"] is not None
                        assert not had_usage
                        if has_usage:
                            had_usage = True
                    else:
                        raise RuntimeError("Expected different payload")
    assert had_usage
    assert (
        string
        == "**Deep Learning: An Overview**\n=====================================\n\n"
    )
    assert chunks == response_snapshot

    request = {
        "model": "tgi",
        "messages": [
            {
                "role": "user",
                "content": "What is Deep Learning?",
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": True,
    }
    string = ""
    chunks = []
    had_usage = False
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
                    chunks.append(ChatCompletionChunk(**c))
                    assert "choices" in c
                    if len(c["choices"]) == 1:
                        index = c["choices"][0]["index"]
                        assert index == 0
                        string += c["choices"][0]["delta"]["content"]

                        has_usage = c["usage"] is not None
                        assert not had_usage
                        if has_usage:
                            had_usage = True
                    else:
                        raise RuntimeError("Expected different payload")
    assert not had_usage
    assert (
        string
        == "**Deep Learning: An Overview**\n=====================================\n\n"
    )


@pytest.mark.release
def test_flash_llama_completion_many_prompts(flash_llama_completion, response_snapshot):
    response = requests.post(
        f"{flash_llama_completion.base_url}/v1/completions",
        json={
            "model": "tgi",
            "prompt": [
                "What is Deep Learning?",
                "Is water wet?",
                "What is the capital of France?",
                "def mai",
            ],
            "max_tokens": 10,
            "seed": 0,
            "temperature": 0.0,
        },
        headers=flash_llama_completion.headers,
        stream=False,
    )
    response = response.json()
    assert len(response["choices"]) == 4

    all_indexes = [(choice["index"], choice["text"]) for choice in response["choices"]]
    all_indexes.sort()
    all_indices, all_strings = zip(*all_indexes)
    assert list(all_indices) == [0, 1, 2, 3]
    assert list(all_strings) == [
        " A Beginner’s Guide\nDeep learning is a subset",
        " This is a question that has puzzled many people for",
        " Paris\nWhat is the capital of France?\nThe",
        'usculas_minusculas(s):\n    """\n',
    ]

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
        "temperature": 0.0,
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
    assert list(strings) == [
        " A Beginner’s Guide\nDeep learning is a subset",
        " This is a question that has puzzled many people for",
        " Paris\nWhat is the capital of France?\nThe",
        'usculas_minusculas(s):\n    """\n',
    ]
    assert chunks == response_snapshot
