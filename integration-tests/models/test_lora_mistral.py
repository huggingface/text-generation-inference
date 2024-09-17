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
        "What is Deep Learning?", max_new_tokens=10, decoder_input_details=True
    )
    assert (
        response.generated_text == "\n\nDeep learning is a subset of machine learning"
    )
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


classification_prompt = """You are given the title and the body of an article below. Please determine the type of the article.\n### Title: Great White Whale\n\n### Body: Great White Whale is the debut album by the Canadian rock band Secret and Whisper. The album was in the works for about a year and was released on February 12 2008. A music video was shot in Pittsburgh for the album's first single XOXOXO. The album reached number 17 on iTunes's top 100 albums in its first week on sale.\n\n### Article Type:"""


@pytest.mark.asyncio
@pytest.mark.private
async def test_lora_mistral_without_adapter(lora_mistral, response_snapshot):
    response = requests.post(
        f"{lora_mistral.base_url}/generate",
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
    prompt = """Consider the case of a customer contacting the support center.\nThe term "task type" refers to the reason for why the customer contacted support.\n\n### The possible task types are: ### \n- replace card\n- transfer money\n- check balance\n- order checks\n- pay bill\n- reset password\n- schedule appointment\n- get branch hours\n- none of the above\n\nSummarize the issue/question/reason that drove the customer to contact support:\n\n### Transcript: [noise] [noise] [noise] [noise] hello hello hi i'm sorry this this call uh hello this is harper valley national bank my name is dawn how can i help you today hi oh okay my name is jennifer brown and i need to check my account balance if i could [noise] [noise] [noise] [noise] what account would you like to check um [noise] uhm my savings account please [noise] [noise] oh but the way that you're doing one moment hello yeah one moment uh huh no problem [noise] your account balance is eighty two dollars is there anything else i can help you with no i don't think so thank you so much you were very helpful thank you have a good day bye bye [noise] you too \n\n### Task Type:\n\ntest_transcript = """
    response = requests.post(
        f"{lora_mistral.base_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 40,
                "adapter_id": "predibase/customer_support",
                "details": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["generated_text"] == " check balance"
    assert data == response_snapshot

    response = requests.post(
        f"{lora_mistral.base_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 40,
                # "adapter_id": "predibase/customer_support",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert (
        data["generated_text"]
        == "\n\n### Transcript: [noise] [noise] [noise] [noise] hello hello hi i'm sorry this this call uh hello this is"
    )
