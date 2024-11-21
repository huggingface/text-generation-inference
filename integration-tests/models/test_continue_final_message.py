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
        == "\nGenerate according to: It is an elephant's one year old baby or a mouse's one year old baby. It is"
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
        == " Shere Kohan's fantastic exploits? written by David Shimomura & illustrated by Sarah Stern\n\nTitle: Elephant"
    )
    assert response == response_snapshot
