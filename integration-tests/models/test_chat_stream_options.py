import pytest


@pytest.fixture(scope="module")
def chat_handle(launcher):
    with launcher(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def chat_client(chat_handle):
    await chat_handle.health(300)
    return chat_handle.client
