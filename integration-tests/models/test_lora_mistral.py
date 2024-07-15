import pytest
import requests


@pytest.fixture(scope="module")
def lora_mistral_handle(launcher):
    with launcher(
        "mistralai/Mistral-7B-v0.1",
        lora_adapters=[
            "predibase/dbpedia",
            "predibase/customer_support",
        ],
        cuda_graphs=[0],
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def lora_mistral(lora_mistral_handle):
    await lora_mistral_handle.health(300)
    return lora_mistral_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_lora_mistral(lora_mistral, response_snapshot):
    response = await lora_mistral.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )
    assert response.details.generated_tokens == 10


classification_prompt = """You are given the title and the body of an article below. Please determine the type of the article.\n### Title: Great White Whale\n\n### Body: Great White Whale is the debut album by the Canadian rock band Secret and Whisper. The album was in the works for about a year and was released on February 12 2008. A music video was shot in Pittsburgh for the album's first single XOXOXO. The album reached number 17 on iTunes's top 100 albums in its first week on sale.\n\n### Article Type:"""


@pytest.mark.asyncio
@pytest.mark.private
async def test_lora_mistral_without_adapter(lora_mistral, response_snapshot):
    response = requests.post(
        f"{lora_mistral.base_url}/generate",
        headers=lora_mistral.headers,
        json={
            "inputs": classification_prompt,
            "parameters": {
                "max_new_tokens": 40,
                "details": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert (
        data["generated_text"]
        == "\n\n### 1. News\n### 2. Blog\n### 3. Article\n### 4. Review\n### 5. Other\n\n\n\n\n\n\n\n\n"
    )
    assert data == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_lora_mistral_with_dbpedia_adapter(lora_mistral, response_snapshot):
    response = requests.post(
        f"{lora_mistral.base_url}/generate",
        headers=lora_mistral.headers,
        json={
            "inputs": classification_prompt,
            "parameters": {
                "max_new_tokens": 40,
                "adapter_id": "predibase/dbpedia",
                "details": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] == "  11"
    assert data == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_lora_mistral_with_customer_support_adapter(
    lora_mistral, response_snapshot
):
    print(lora_mistral.base_url)
    print(lora_mistral.headers)
    response = requests.post(
        f"{lora_mistral.base_url}/generate",
        headers=lora_mistral.headers,
        json={
            "inputs": "What are 3 unique words that describe you?",
            "parameters": {
                "max_new_tokens": 40,
                "adapter_id": "predibase/customer_support",
                "details": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert (
        data["generated_text"]
        == "\n\nI’m not sure if I can come up with 3 unique words that describe me, but I’ll try.\n\n1. Creative\n2. Funny\n3."
    )
    assert data == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_lora_mistral_without_customer_support_adapter(
    lora_mistral, response_snapshot
):
    response = requests.post(
        f"{lora_mistral.base_url}/generate",
        headers=lora_mistral.headers,
        json={
            "inputs": "What are 3 unique words that describe you?",
            "parameters": {
                "max_new_tokens": 40,
                "details": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert (
        data["generated_text"]
        == "\n\nI’m a very passionate person. I’m very driven. I’m very determined.\n\nWhat is your favorite thing about being a teacher?\n\nI love the fact"
    )
    assert data == response_snapshot
