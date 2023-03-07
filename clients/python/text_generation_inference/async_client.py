import json

from aiohttp import ClientSession, ClientTimeout
from pydantic import ValidationError
from typing import Optional

from text_generation_inference.types import StreamResponse, ErrorModel, Response


class AsyncClient:
    def __init__(self, model_id: str, token: Optional[str] = None, timeout: int = 10):
        headers = {}
        if token is not None:
            headers = {"Authorization": f"Bearer {token}"}
        self.model_id = model_id

        self.session = ClientSession(headers=headers, timeout=ClientTimeout(timeout * 60))

    async def generate(self):
        async with self.session.post(f"https://api-inference.huggingface.co/models/{self.model_id}",
                                     json={"inputs": "test", "stream": True}) as resp:
            if resp.status != 200:
                error = ErrorModel(**await resp.json())
                raise error.to_exception()
            return Response(**await resp.json())

    async def generate_stream(self):
        async with self.session.post(f"https://api-inference.huggingface.co/models/{self.model_id}",
                                     json={"inputs": "test", "stream": True}) as resp:
            if resp.status != 200:
                error = ErrorModel(**await resp.json())
                raise error.to_exception()

            async for byte_payload in resp.content:
                if byte_payload == b"\n":
                    continue

                payload = byte_payload.decode("utf-8")

                if payload.startswith("data:"):
                    json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                    try:
                        response = StreamResponse(**json_payload)
                    except ValidationError:
                        error = ErrorModel(**json_payload)
                        raise error.to_exception()
                    yield response.token

    def __del__(self):
        self.session.close()


async def main():
    client = AsyncClient("bigscience/bloomz")
    async for token in client.generate_stream():
        print(token)

    print(await client.generate())


import asyncio

asyncio.run(main())
