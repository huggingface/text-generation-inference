import pytest
import requests
from openai import OpenAI
from huggingface_hub import InferenceClient


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
    client = InferenceClient(base_url=f"{flash_llama_completion.base_url}/v1")
    stream = client.chat_completion(
        model="tgi",
        messages=[
            {
                "role": "user",
                "content": "What is Deep Learning?",
            }
        ],
        max_tokens=10,
        temperature=0.0,
        stream_options={"include_usage": True},
        stream=True,
    )
    string = ""
    chunks = []
    had_usage = False
    for chunk in stream:
        # remove "data:"
        chunks.append(chunk)
        if len(chunk.choices) == 1:
            index = chunk.choices[0].index
            assert index == 0
            string += chunk.choices[0].delta.content
        if chunk.usage:
            assert not had_usage
            had_usage = True

    assert had_usage
    assert (
        string
        == "**Deep Learning: An Overview**\n=====================================\n\n"
    )
    assert chunks == response_snapshot

    stream = client.chat_completion(
        model="tgi",
        messages=[
            {
                "role": "user",
                "content": "What is Deep Learning?",
            }
        ],
        max_tokens=10,
        temperature=0.0,
        # No usage
        # stream_options={"include_usage": True},
        stream=True,
    )
    string = ""
    chunks = []
    had_usage = False
    for chunk in stream:
        chunks.append(chunk)
        assert chunk.usage is None
        assert len(chunk.choices) == 1
        assert chunk.choices[0].index == 0
        string += chunk.choices[0].delta.content
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
    client = OpenAI(api_key="xx", base_url=f"{flash_llama_completion.base_url}/v1")
    stream = client.completions.create(
        model="tgi",
        prompt=[
            "What is Deep Learning?",
            "Is water wet?",
            "What is the capital of France?",
            "def mai",
        ],
        max_tokens=10,
        seed=0,
        temperature=0.0,
        stream=True,
    )

    strings = [""] * 4
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        index = chunk.choices[0].index
        assert 0 <= index <= 4
        strings[index] += chunk.choices[0].text

    assert list(strings) == [
        " A Beginner’s Guide\nDeep learning is a subset",
        " This is a question that has puzzled many people for",
        " Paris\nWhat is the capital of France?\nThe",
        'usculas_minusculas(s):\n    """\n',
    ]
    assert chunks == response_snapshot


@pytest.mark.release
async def test_chat_openai_usage(flash_llama_completion, response_snapshot):
    client = OpenAI(api_key="xx", base_url=f"{flash_llama_completion.base_url}/v1")

    stream = client.chat.completions.create(
        model="tgi",
        messages=[{"role": "user", "content": "Say 'OK!'"}],
        stream=True,
        max_tokens=10,
        seed=42,
        stream_options={"include_usage": True},
    )

    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    for chunk in chunks[:-1]:
        assert chunk.usage is None
    for chunk in chunks[-1:]:
        assert chunk.usage is not None

    assert chunks == response_snapshot


@pytest.mark.release
async def test_chat_openai_nousage(flash_llama_completion, response_snapshot):
    client = OpenAI(api_key="xx", base_url=f"{flash_llama_completion.base_url}/v1")

    stream = client.chat.completions.create(
        model="tgi",
        messages=[{"role": "user", "content": "Say 'OK!'"}],
        stream=True,
        max_tokens=10,
        seed=42,
        stream_options={"include_usage": False},
    )

    chunks = []
    for chunk in stream:
        assert chunk.usage is None
        chunks.append(chunk)

    assert chunks == response_snapshot


@pytest.mark.release
async def test_chat_hfhub_usage(flash_llama_completion, response_snapshot):
    client = InferenceClient(base_url=f"{flash_llama_completion.base_url}/v1")
    stream = client.chat_completion(
        model="tgi",
        messages=[{"role": "user", "content": "Say 'OK!'"}],
        stream=True,
        max_tokens=10,
        seed=42,
        stream_options={"include_usage": True},
    )

    chunks = []
    for chunk in stream:
        chunks.append(chunk)

    for chunk in chunks[:-1]:
        assert chunk.usage is None
    for chunk in chunks[-1:]:
        assert chunk.usage is not None

    assert chunks == response_snapshot


@pytest.mark.release
async def test_chat_hfhub_nousage(flash_llama_completion, response_snapshot):
    client = InferenceClient(base_url=f"{flash_llama_completion.base_url}/v1")
    stream = client.chat_completion(
        model="tgi",
        messages=[{"role": "user", "content": "Say 'OK!'"}],
        stream=True,
        max_tokens=10,
        seed=42,
        stream_options={"include_usage": False},
    )

    chunks = []
    for chunk in stream:
        assert chunk.usage is None
        chunks.append(chunk)

    assert chunks == response_snapshot
