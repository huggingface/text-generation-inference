import pytest
import requests
from pydantic import BaseModel
from typing import List


@pytest.fixture(scope="module")
def llama_grammar_handle(launcher):
    with launcher(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_shard=1,
        disable_grammar_support=False,
        use_flash_attention=False,
        max_batch_prefill_tokens=3000,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def llama_grammar(llama_grammar_handle):
    await llama_grammar_handle.health(300)
    return llama_grammar_handle.client


@pytest.mark.asyncio
async def test_grammar_response_format_llama_json(llama_grammar, response_snapshot):

    class Weather(BaseModel):
        unit: str
        temperature: List[int]

    # send the request
    response = requests.post(
        f"{llama_grammar.base_url}/v1/chat/completions",
        headers=llama_grammar.headers,
        json={
            "model": "tgi",
            "messages": [
                {
                    "role": "system",
                    "content": f"Respond to the users questions and answer them in the following format: {Weather.schema()}",
                },
                {
                    "role": "user",
                    "content": "What's the weather like the next 3 days in San Francisco, CA?",
                },
            ],
            "seed": 42,
            "max_tokens": 500,
            "response_format": {"type": "json_object", "value": Weather.schema()},
        },
    )

    chat_completion = response.json()
    called = chat_completion["choices"][0]["message"]["content"]

    assert response.status_code == 200
    assert (
        called
        == '{\n  "temperature": [\n    35,\n    34,\n    36\n  ],\n  "unit": "°c"\n}'
    )
    assert chat_completion == response_snapshot


@pytest.mark.asyncio
async def test_grammar_response_format_llama_error_if_tools_not_installed(
    llama_grammar,
):
    class Weather(BaseModel):
        unit: str
        temperature: List[int]

    # send the request
    response = requests.post(
        f"{llama_grammar.base_url}/v1/chat/completions",
        headers=llama_grammar.headers,
        json={
            "model": "tgi",
            "messages": [
                {
                    "role": "system",
                    "content": f"Respond to the users questions and answer them in the following format: {Weather.schema()}",
                },
                {
                    "role": "user",
                    "content": "What's the weather like the next 3 days in San Francisco, CA?",
                },
            ],
            "seed": 42,
            "max_tokens": 500,
            "tools": [],
            "response_format": {"type": "json_object", "value": Weather.schema()},
        },
    )

    # 422 means the server was unable to process the request because it contains invalid data.
    assert response.status_code == 422
    assert response.json() == {
        "error": "Grammar and tools are mutually exclusive",
        "error_type": "grammar and tools",
    }
