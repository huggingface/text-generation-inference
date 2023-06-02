import pytest


@pytest.fixture(scope="module")
def mt0_base_handle(launcher):
    with launcher("bigscience/mt0-base") as handle:
        yield handle


@pytest.fixture(scope="module")
async def mt0_base(mt0_base_handle):
    await mt0_base_handle.health(300)
    return mt0_base_handle.client


@pytest.mark.asyncio
async def test_mt0_base(mt0_base, response_snapshot):
    response = await mt0_base.generate(
        "Why is the sky blue?",
        max_new_tokens=10,
        top_p=0.9,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 5
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_mt0_base_all_params(mt0_base, response_snapshot):
    response = await mt0_base.generate(
        "Why is the sky blue?",
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

    assert response.details.generated_tokens == 9
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_mt0_base_load(mt0_base, generate_load, response_snapshot):
    responses = await generate_load(
        mt0_base,
        "Why is the sky blue?",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
