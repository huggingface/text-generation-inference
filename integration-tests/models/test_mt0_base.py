import pytest

from utils import health_check


@pytest.fixture(scope="module")
def mt0_base(launcher):
    with launcher("bigscience/mt0-base") as client:
        yield client


@pytest.mark.asyncio
async def test_mt0_base(mt0_base, snapshot_test):
    await health_check(mt0_base, 60)

    response = await mt0_base.generate(
        "Why is the sky blue?",
        max_new_tokens=10,
        top_p=0.9,
        seed=0,
    )

    assert response.details.generated_tokens == 5
    assert snapshot_test(response)


@pytest.mark.asyncio
async def test_mt0_base_all_params(mt0_base, snapshot_test):
    await health_check(mt0_base, 60)

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
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert snapshot_test(response)


@pytest.mark.asyncio
async def test_mt0_base_load(mt0_base, generate_load, snapshot_test):
    await health_check(mt0_base, 60)

    responses = await generate_load(
        mt0_base,
        "Why is the sky blue?",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4

    assert snapshot_test(responses)
