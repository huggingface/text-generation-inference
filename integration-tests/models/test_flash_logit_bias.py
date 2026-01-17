import pytest


@pytest.fixture(scope="module")
def logit_bias_model_handle(launcher):
    with launcher("Qwen/Qwen2-VL-2B-Instruct") as handle:
        yield handle


@pytest.fixture(scope="module")
async def logit_bias_model(logit_bias_model_handle):
    await logit_bias_model_handle.health(300)
    return logit_bias_model_handle.client


@pytest.mark.private
async def test_logit_bias_english_to_spanish(logit_bias_model, response_snapshot):
    """Test that setting negative bias on English tokens forces output to be in Spanish"""
    response = await logit_bias_model.chat(
        seed=42,
        max_tokens=10,
        logit_bias={"9707": -100},  # Bias against 'Hello' token
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "say Hello"},
                ],
            },
        ],
    )
    assert "¡Hola!" in response.choices[0].message.content
    assert "Hello" not in response.choices[0].message.content
    assert response == response_snapshot


@pytest.mark.private
async def test_logit_bias_baseline(logit_bias_model, response_snapshot):
    """Test baseline behavior without logit bias for comparison"""
    response = await logit_bias_model.chat(
        seed=42,
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "say Hello"},
                ],
            },
        ],
    )
    assert "Hello" in response.choices[0].message.content
    assert response == response_snapshot


@pytest.mark.private
async def test_logit_bias_multiple_tokens(logit_bias_model, response_snapshot):
    """Test applying bias to multiple tokens simultaneously"""
    response = await logit_bias_model.chat(
        seed=42,
        max_tokens=15,
        logit_bias={
            "9707": -100,  # Bias against 'Hello' token
            "2880": -100,  # Bias against 'hi' token
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Give me a one-word greeting"},
                ],
            },
        ],
    )
    assert "Hello" not in response.choices[0].message.content.lower()
    assert "hi" not in response.choices[0].message.content.lower()
    assert response == response_snapshot


@pytest.mark.private
async def test_logit_bias_streaming(logit_bias_model, response_snapshot):
    """Test logit bias works correctly with streaming enabled"""
    responses = await logit_bias_model.chat(
        seed=42,
        max_tokens=10,
        logit_bias={"9707": -100},  # Bias against 'Hello' token
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "say Hello"},
                ],
            },
        ],
        stream=True,
    )

    count = 0
    generated = ""
    last_response = None

    async for response in responses:
        count += 1
        generated += response.choices[0].delta.content
        last_response = response

    assert "¡Hola!" in generated
    assert "Hello" not in generated
    assert last_response == response_snapshot
