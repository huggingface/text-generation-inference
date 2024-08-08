import pytest


@pytest.fixture(scope="module")
def opt_sharded_handle(launcher):
    with launcher("facebook/opt-6.7b", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def opt_sharded(opt_sharded_handle):
    await opt_sharded_handle.health(300)
    return opt_sharded_handle.client


@pytest.mark.release
@pytest.mark.asyncio
async def test_opt(opt_sharded):
    pass
