import pytest
import json
import asyncio
from aiohttp import ClientSession


@pytest.fixture(scope="module")
def chat_handle(launcher):
    with launcher(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def chat_client(chat_handle):
    await chat_handle.health(300)
    return chat_handle.client


@pytest.mark.release
async def test_chat_stream_options_include_usage(chat_client, response_snapshot):
    request = {
        "model": "tgi",
        "messages": [{"role": "user", "content": "say 'OK!'"}],
        "max_tokens": 10,
        "stream": True,
        "seed": 42,
        "stream_options": {"include_usage": True},
    }

    url = f"{chat_client.base_url}/v1/chat/completions"

    chunks = []
    async with ClientSession(headers=chat_client.headers) as session:
        async with session.post(url, json=request) as response:
            async for chunk in response.content.iter_any():
                for c in chunk.decode().split("\n\n"):
                    c = c.replace("data: ", "")
                    if not c:
                        continue
                    if c == "[DONE]":
                        break
                    c = json.loads(c)
                    chunks.append(c)
                    assert "choices" in c

                    # keep updating the final chunk prior to "[DONE]"
                    final_chunk = c

    assert final_chunk["usage"] is not None, "Usage information missing in final chunk"
    assert len(final_chunk["choices"]) == 0, "Choices should be empty in final chunk"
    assert response.status == 200
    assert chunks == response_snapshot


@pytest.mark.release
async def test_chat_stream_options(chat_client, response_snapshot):
    request = {
        "model": "tgi",
        "messages": [{"role": "user", "content": "say 'OK!'"}],
        "max_tokens": 10,
        "stream": True,
        "seed": 42,
        "stream_options": {"include_usage": False},
    }

    url = f"{chat_client.base_url}/v1/chat/completions"

    chunks = []
    async with ClientSession(headers=chat_client.headers) as session:
        async with session.post(url, json=request) as response:
            async for chunk in response.content.iter_any():
                for c in chunk.decode().split("\n\n"):
                    c = c.replace("data: ", "")
                    if not c:
                        continue
                    if c == "[DONE]":
                        break
                    c = json.loads(c)
                    chunks.append(c)
                    assert "choices" in c

                    # keep updating the final chunk prior to "[DONE]"
                    final_chunk = c

    assert final_chunk["usage"] is None, "Usage information should not be present"
    assert len(final_chunk["choices"]) == 1, "Choices should be present in final chunk"

    assert response.status == 200
    assert chunks == response_snapshot
