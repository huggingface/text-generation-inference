import pytest


@pytest.fixture(scope="module")
def flash_llama_awq_handle_sharded(launcher):
    with launcher(
        "abhinavkulkarni/codellama-CodeLlama-7b-Python-hf-w4-g128-awq",
        num_shard=2,
        quantize="awq",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_awq_sharded(flash_llama_awq_handle_sharded):
    await flash_llama_awq_handle_sharded.health(300)
    return flash_llama_awq_handle_sharded.client


@pytest.mark.asyncio
async def test_flash_llama_awq_sharded(flash_llama_awq_sharded, response_snapshot):
    response = await flash_llama_awq_sharded.generate(
        "What is Deep Learning?", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert (
        response.generated_text
        == "\nWhat is the difference between Deep Learning and Machine"
    )
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_llama_awq_load_sharded(
    flash_llama_awq_sharded, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_llama_awq_sharded, "What is Deep Learning?", max_new_tokens=10, n=4
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
