import pytest


@pytest.fixture(scope="module")
def flash_falcon_handle(launcher):
    with launcher("tiiuae/falcon-7b", trust_remote_code=True) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_falcon(flash_falcon_handle):
    await flash_falcon_handle.health(300)
    return flash_falcon_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_falcon(flash_falcon, response_snapshot):
    response = await flash_falcon.generate(
        "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_falcon_all_params(flash_falcon, response_snapshot):
    response = await flash_falcon.generate(
        "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
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
async def test_flash_falcon_load(flash_falcon, generate_load, response_snapshot):
    responses = await generate_load(
        flash_falcon,
        "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
