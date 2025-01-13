import pytest


@pytest.fixture(scope="module")
def flash_idefics3_next_handle(launcher):
    with launcher("HuggingFaceM4/Idefics3-8B-Llama3") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_idefics3_next(flash_idefics3_next_handle):
    await flash_idefics3_next_handle.health(300)
    return flash_idefics3_next_handle.client


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
        response.generated_text == " There is a statue in the image."
    ), f"{repr(response.generated_text)}"
    assert response.details.generated_tokens == 9
    assert response == response_snapshot
