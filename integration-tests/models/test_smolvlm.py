import pytest


@pytest.fixture(scope="module")
def flash_smolvlm_next_handle(launcher):
    with launcher("HuggingFaceTB/SmolVLM-Instruct") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_smolvlm_next(flash_smolvlm_next_handle):
    await flash_smolvlm_next_handle.health(300)
    return flash_smolvlm_next_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_smolvlm_next_simple_url(flash_smolvlm_next, response_snapshot):
    ny_skyline = "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"
    query = "What is in this image?"
    response = await flash_smolvlm_next.generate(
        f"<|begin_of_text|><|begin_of_text|>User:![]({ny_skyline}){query}<end_of_utterance>\nAssistant:",
        max_new_tokens=10,
        seed=1337,
    )
    print(response)
    assert (
        response.generated_text == " A bee on a pink flower."
    ), f"{repr(response.generated_text)}"
    assert response.details.generated_tokens == 8
    assert response == response_snapshot
