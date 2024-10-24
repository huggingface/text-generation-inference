import pytest
import base64


def get_chicken():
    with open("integration-tests/images/chicken_on_money.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return f"data:image/png;base64,{encoded_string.decode('utf-8')}"


@pytest.fixture(scope="module")
def flash_idefics3_next_handle(launcher):
    with launcher(
        "HuggingFaceM4/Idefics3-8B-Llama3",
        max_total_tokens=3000,
        max_batch_prefill_tokens=2501,
        max_input_tokens=2500,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_idefics3_next(flash_idefics3_next_handle):
    await flash_idefics3_next_handle.health(300)
    return flash_idefics3_next_handle.client


# TODO: dont skip when token issue is resolved
@pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_idefics3_next_simple_base64(
    flash_idefics3_next, response_snapshot
):
    chicken = get_chicken()
    query = "Write me a short story"
    response = await flash_idefics3_next.generate(
        f"<|begin_of_text|><|begin_of_text|>User:![]({chicken}){query}<end_of_utterance>\nAssistant:",
        max_new_tokens=10,
    )
    assert (
        response.generated_text == " A chicken is sitting on a pile of money."
    ), f"{repr(response.generated_text)}"
    # assert response.details.generated_tokens == 10
    # assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_idefics3_next_simple_url(flash_idefics3_next, response_snapshot):
    ny_skyline = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    query = "What is in this image?"
    response = await flash_idefics3_next.generate(
        f"<|begin_of_text|><|begin_of_text|>User:![]({ny_skyline}){query}<end_of_utterance>\nAssistant:",
        max_new_tokens=10,
        seed=1337,
    )
    print(response)
    assert (
        response.generated_text
        == " The image depicts the Statue of Liberty, a colossal"
    ), f"{repr(response.generated_text)}"
    assert response.details.generated_tokens == 10
    assert response == response_snapshot
