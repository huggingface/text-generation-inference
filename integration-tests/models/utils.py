import time

from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
from text_generation import AsyncClient


async def health_check(client: AsyncClient, timeout: int = 60):
    assert timeout > 0
    for _ in range(timeout):
        try:
            await client.generate("test")
            return
        except (ClientConnectorError, ClientOSError, ServerDisconnectedError) as e:
            time.sleep(1)
    raise e
