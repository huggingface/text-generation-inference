import pytest
import base64


@pytest.fixture(scope="module")
def idefics_handle(launcher):
    with launcher(
        "HuggingFaceM4/idefics-9b-instruct", num_shard=2, dtype="float16"
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def idefics(idefics_handle):
    await idefics_handle.health(300)
    return idefics_handle.client


# TODO fix the server parsser to count inline image tokens correctly
def get_chicken():
    with open("integration-tests/images/chicken_on_money.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return f"data:image/png;base64,{encoded_string.decode('utf-8')}"


def get_cow_beach():
    with open("integration-tests/images/cow_beach.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return f"data:image/png;base64,{encoded_string.decode('utf-8')}"


@pytest.mark.asyncio
async def test_idefics(idefics, response_snapshot):
    chicken = get_chicken()
    response = await idefics.generate(
        f"User:![]({chicken})Can you tell me a very short story based on the image?",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert response.details.generated_tokens == 10
    assert (
        response.generated_text == " \nAssistant: A rooster stands"
    ), f"{repr(response.generated_text)}"
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_idefics_two_images(idefics, response_snapshot):
    chicken = get_chicken()
    cow_beach = get_cow_beach()
    response = await idefics.generate(
        f"User:![]({chicken})![]({cow_beach})Where are the cow and chicken?<end_of_utterance> \nAssistant:",
        max_new_tokens=20,
    )
    assert (
        response.generated_text == " The cow and chicken are on a beach."
    ), f"{repr(response.generated_text)}"
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
async def test_idefics_load(idefics, generate_load, response_snapshot):
    chicken = get_chicken()
    responses = await generate_load(
        idefics,
        f"User:![]({chicken})Can you tell me a very short story based on the image?",
        max_new_tokens=10,
        n=4,
    )

    generated_texts = [r.generated_text for r in responses]

    assert generated_texts[0] == " \nAssistant: A rooster stands"
    assert len(generated_texts) == 4
    assert generated_texts, all(
        [text == generated_texts[0] for text in generated_texts]
    )

    assert responses == response_snapshot
