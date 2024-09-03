import pytest


@pytest.fixture(scope="module")
def flash_phi35_moe_chat_handle(launcher):
    with launcher(
        "microsoft/Phi-3.5-MoE-instruct", num_shard=4, cuda_graphs=[1, 2]
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_phi35_moe_chat(flash_phi35_moe_chat_handle):
    await flash_phi35_moe_chat_handle.health(300)
    return flash_phi35_moe_chat_handle.client


@pytest.mark.private
async def test_flash_phi35_moe_simple(flash_phi35_moe_chat, response_snapshot):
    response = await flash_phi35_moe_chat.chat(
        max_tokens=100,
        seed=1337,
        messages=[
            {
                "role": "system",
                "content": "Youre a helpful assistant! Answer the users question best you can.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Brooklyn, New York?",
            },
        ],
    )

    assert (
        response.choices[0].message.content
        == "I'm an AI unable to provide real-time data, but I can guide you on how to find current weather conditions in Brooklyn, New York. You can check websites like weather.com or accuweather.com, or use apps like The Weather Channel or AccuWeather on your smartphone. Alternatively, you can ask your voice assistant like Google Assistant or Siri for real-time updates.\n\nFor your information, I hope you'll have a"
    )
    assert response == response_snapshot
