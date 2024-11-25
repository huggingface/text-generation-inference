import pytest
import requests


@pytest.fixture(scope="module")
def llama_continue_final_message_handle(launcher):
    with launcher("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as handle:
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
                {"role": "user", "content": "Which is bigger an elephant or a mouse?"},
            ],
            "max_tokens": 30,
            "stream": False,
            "seed": 1337,
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
        == "Both an elephant and a mouse are mammals. However, the differences between elephants and mice are:\n\n1"
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
                {"role": "user", "content": "Which is bigger an elephant or a mouse?"},
                {
                    "role": "assistant",
                    "content": "the elephant, but have you heard about",
                },
            ],
            "max_tokens": 30,
            "stream": False,
            "seed": 1337,
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
        == " the royal mouse? It is a little more slender and only weighs around 1.5 pounds for males and 1.3 pounds"
    )
    assert response == response_snapshot
