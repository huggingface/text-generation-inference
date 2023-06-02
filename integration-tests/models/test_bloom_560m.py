import pytest


@pytest.fixture(scope="module")
def bloom_560_handle(launcher):
    with launcher("bigscience/bloom-560m") as handle:
        yield handle


@pytest.fixture(scope="module")
async def bloom_560(bloom_560_handle):
    await bloom_560_handle.health(240)
    return bloom_560_handle.client


@pytest.mark.asyncio
async def test_bloom_560m(bloom_560, response_snapshot):
    response = await bloom_560.generate(
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        top_p=0.9,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_bloom_560m_all_params(bloom_560, response_snapshot):
    response = await bloom_560.generate(
        "Pour déguster un ortolan, il faut tout d'abord",
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
async def test_bloom_560m_load(bloom_560, generate_load, response_snapshot):
    responses = await generate_load(
        bloom_560,
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
