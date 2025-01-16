import pytest
import requests


@pytest.fixture(scope="module")
def flash_starcoder2_handle(launcher):
    with launcher(
        "bigcode/starcoder2-3b", lora_adapters=["smangrul/starcoder-3b-hugcoder"]
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_starcoder2(flash_starcoder2_handle):
    await flash_starcoder2_handle.health(300)
    return flash_starcoder2_handle.client


@pytest.mark.asyncio
async def test_flash_starcoder2(flash_starcoder2, response_snapshot):
    response = await flash_starcoder2.generate(
        "def print_hello", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_starcoder2_default_params(flash_starcoder2, response_snapshot):
    response = await flash_starcoder2.generate(
        "who are you?",
        max_new_tokens=60,
        temperature=0.2,
        top_p=0.95,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 60
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_starcoder2_load(
    flash_starcoder2, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_starcoder2, "who are you?", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot


@pytest.mark.asyncio
async def test_flash_starcoder2_with_hugcode_adapter(
    flash_starcoder2, response_snapshot
):
    response = requests.post(
        f"{flash_starcoder2.base_url}/generate",
        headers=flash_starcoder2.headers,
        json={
            "inputs": "def print_hello",
            "parameters": {
                "max_new_tokens": 10,
                "adapter_id": "smangrul/starcoder-3b-hugcoder",
                "details": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] == '_world():\n    print("Hello World!")\n'

    assert data == response_snapshot
