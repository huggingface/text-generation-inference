import pytest
import base64


# TODO fix the server parsser to count inline image tokens correctly
def get_chicken():
    with open("integration-tests/images/chicken_on_money.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return f"data:image/png;base64,{encoded_string.decode('utf-8')}"


def get_cow_beach():
    with open("integration-tests/images/cow_beach.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return f"data:image/png;base64,{encoded_string.decode('utf-8')}"


@pytest.fixture(scope="module")
def flash_idefics2_next_handle(launcher):
    with launcher(
        "HuggingFaceM4/idefics2-8b",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_idefics2_next(flash_idefics2_next_handle):
    await flash_idefics2_next_handle.health(300)
    return flash_idefics2_next_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_idefics2_next_simple(flash_idefics2_next, response_snapshot):
    chicken = get_chicken()
    response = await flash_idefics2_next.generate(
        f"User:![]({chicken})Write me a short story<end_of_utterance> \nAssistant:",
        max_new_tokens=10,
    )
    assert (
        response.generated_text == " A chicken is sitting on a pile of money."
    ), f"{repr(response.generated_text)}"
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_idefics2_two_images(flash_idefics2_next, response_snapshot):
    chicken = get_chicken()
    cow_beach = get_cow_beach()
    response = await flash_idefics2_next.generate(
        f"User:![]({chicken})![]({cow_beach})Where are the cow and chicken?<end_of_utterance> \nAssistant:",
        max_new_tokens=20,
    )
    assert (
        response.generated_text
        == " The cow is standing on the beach and the chicken is sitting on a pile of money."
    ), f"{repr(response.generated_text)}"
    assert response.details.generated_tokens == 19
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_idefics2_next_all_params(flash_idefics2_next, response_snapshot):
    response = await flash_idefics2_next.generate(
        "Test request",
        max_new_tokens=10,
        repetition_penalty=1.2,
        return_full_text=True,
        stop_sequences=["test"],
        temperature=0.5,
        top_p=0.9,
        top_k=10,
        truncate=5,
        typical_p=0.9,
        watermark=True,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_idefics2_next_load(
    flash_idefics2_next, generate_load, response_snapshot
):
    chicken = get_chicken()
    responses = await generate_load(
        flash_idefics2_next,
        f"User:![]({chicken})Write me a short story<end_of_utterance> \nAssistant:",
        max_new_tokens=10,
        n=4,
    )
    generated_texts = [r.generated_text for r in responses]
    assert generated_texts[0] == " A chicken is sitting on a pile of money."
    assert len(generated_texts) == 4
    assert all([r.generated_text == generated_texts[0] for r in responses])

    assert responses == response_snapshot
