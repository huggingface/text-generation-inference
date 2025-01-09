import pytest
import json
import requests


@pytest.fixture(scope="module")
def qwen2_vl_handle(launcher):
    with launcher(
        "Qwen/Qwen2-VL-7B-Instruct",
        max_input_length=10_000,
        max_batch_prefill_tokens=10_000,
        max_total_tokens=10_001,
        cuda_graphs=[0],
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def qwen2_vl(qwen2_vl_handle):
    await qwen2_vl_handle.health(300)
    return qwen2_vl_handle.client


@pytest.mark.asyncio
async def test_qwen2_vl_simpl(qwen2_vl, response_snapshot):
    responses = requests.post(
        f"{qwen2_vl.base_url}/v1/chat/completions",
        headers=qwen2_vl.headers,
        json={
            "model": "tgi",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this video.",
                        },
                    ],
                },
            ],
            "seed": 42,
            "max_tokens": 100,
            "stream": True,
        },
    )

    # iterate over the response in chunks
    count = 0
    full_text = ""
    last_response = None
    for chunk in responses.iter_content(chunk_size=1024):
        if chunk:
            count += 1
            # remove the "data: " prefix, trailing newline, and split the chunk into individual lines
            lines = chunk.decode("utf-8").replace("data: ", "").rstrip("\n").split("\n")
            for line in lines:
                if line == "[DONE]":
                    break
                print("=", line)
                try:
                    response = json.loads(line)
                    # print(response)
                    last_response = response
                    full_text += response["choices"][0]["delta"]["content"]
                except json.JSONDecodeError:
                    pass

    assert last_response == response_snapshot
