import pytest
import requests


@pytest.fixture(scope="module")
def llama_continue_final_message_handle(launcher):
    with launcher(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_shard=1,
        disable_grammar_support=False,
        use_flash_attention=False,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def llama_continue_final_message(llama_continue_final_message_handle):
    await llama_continue_final_message_handle.health(300)
    return llama_continue_final_message_handle.client


def test_llama_completion_single_prompt(
    llama_continue_final_message, response_snapshot
):
    response = requests.post(
        f"{llama_continue_final_message.base_url}/v1/chat/completions",
        json={
            "model": "tgi",
            "messages": [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {"role": "assistant", "content": "assistant message"},
            ],
            "max_tokens": 30,
            "stream": False,
            "seed": 1337,
            "continue_final_message": False,
        },
        headers=llama_continue_final_message.headers,
        stream=False,
    )
    response = response.json()
    print(response)
    assert len(response["choices"]) == 1
    content = response["choices"][0]["message"]["content"]
    assert (
        content
        == "Hi, I hope this is the right place for your written question. Please provide the maximum possible length to help me complete the message for you! Based"
    )
    assert response == response_snapshot


def test_llama_completion_single_prompt_continue(
    llama_continue_final_message, response_snapshot
):
    response = requests.post(
        f"{llama_continue_final_message.base_url}/v1/chat/completions",
        json={
            "model": "tgi",
            "messages": [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {"role": "assistant", "content": "assistant message"},
            ],
            "max_tokens": 30,
            "stream": False,
            "seed": 1337,
            "continue_final_message": True,
        },
        headers=llama_continue_final_message.headers,
        stream=False,
    )
    response = response.json()
    print(response)
    assert len(response["choices"]) == 1
    content = response["choices"][0]["message"]["content"]
    assert (
        content
        == ": Thanks for the awesome slides, they were just what we needed to produce the presentation we needed to deliver for our company's budgeting system"
    )
    assert response == response_snapshot
