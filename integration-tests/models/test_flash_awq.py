import pytest


@pytest.fixture(scope="module")
def flash_llama_awq_handle(launcher):
    with launcher(
        "abhinavkulkarni/codellama-CodeLlama-7b-Python-hf-w4-g128-awq",
        num_shard=1,
        quantize="awq",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_awq(flash_llama_awq_handle):
    await flash_llama_awq_handle.health(300)
    return flash_llama_awq_handle.client


@pytest.mark.release
@pytest.mark.asyncio
async def test_flash_llama_awq(flash_llama_awq, response_snapshot):
    response = await flash_llama_awq.generate(
        "What is Deep Learning?", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert (
        response.generated_text
        == "\nWhat is the difference between Deep Learning and Machine"
    )
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
async def test_flash_llama_awq_all_params(flash_llama_awq, response_snapshot):
    response = await flash_llama_awq.generate(
        "What is Deep Learning?",
        max_new_tokens=10,
        repetition_penalty=1.2,
        return_full_text=True,
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


@pytest.mark.release
@pytest.mark.asyncio
async def test_flash_llama_awq_load(flash_llama_awq, generate_load, response_snapshot):
    responses = await generate_load(
        flash_llama_awq, "What is Deep Learning?", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all(
        [
            r.generated_text
            == "\nWhat is the difference between Deep Learning and Machine"
            for r in responses
        ]
    )

    assert responses == response_snapshot
