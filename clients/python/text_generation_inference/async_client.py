import json
import os

from aiohttp import ClientSession, ClientTimeout
from pydantic import ValidationError
from typing import Dict, Optional, List, AsyncIterator

from text_generation_inference import SUPPORTED_MODELS
from text_generation_inference.types import (
    StreamResponse,
    Response,
    Request,
    Parameters,
)
from text_generation_inference.errors import parse_error, NotSupportedError

INFERENCE_ENDPOINT = os.environ.get(
    "HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co"
)


class AsyncClient:
    def __init__(
            self, base_url: str, headers: Dict[str, str] = None, timeout: int = 10
    ):
        self.base_url = base_url
        self.headers = headers
        self.timeout = ClientTimeout(timeout * 60)

    async def generate(
            self,
            prompt: str,
            do_sample: bool = False,
            max_new_tokens: int = 20,
            repetition_penalty: Optional[float] = None,
            return_full_text: bool = False,
            seed: Optional[int] = None,
            stop: Optional[List[str]] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            watermark: bool = False,
    ) -> Response:
        parameters = Parameters(
            details=True,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop if stop is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            watermark=watermark,
        )
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        async with ClientSession(headers=self.headers, timeout=self.timeout) as session:
            async with session.post(self.base_url, json=request.dict()) as resp:
                payload = await resp.json()
                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return Response(**payload[0])

    async def generate_stream(
            self,
            prompt: str,
            do_sample: bool = False,
            max_new_tokens: int = 20,
            repetition_penalty: Optional[float] = None,
            return_full_text: bool = False,
            seed: Optional[int] = None,
            stop: Optional[List[str]] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            watermark: bool = False,
    ) -> AsyncIterator[StreamResponse]:
        parameters = Parameters(
            details=True,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop if stop is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            watermark=watermark,
        )
        request = Request(inputs=prompt, stream=True, parameters=parameters)

        async with ClientSession(headers=self.headers, timeout=self.timeout) as session:
            async with session.post(self.base_url, json=request.dict()) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json())

                async for byte_payload in resp.content:
                    if byte_payload == b"\n":
                        continue

                    payload = byte_payload.decode("utf-8")

                    if payload.startswith("data:"):
                        json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                        try:
                            response = StreamResponse(**json_payload)
                        except ValidationError:
                            raise parse_error(resp.status, json_payload)
                        yield response


class APIInferenceAsyncClient(AsyncClient):
    def __init__(self, model_id: str, token: Optional[str] = None, timeout: int = 10):
        # Text Generation Inference client only supports a subset of the available hub models
        if model_id not in SUPPORTED_MODELS:
            raise NotSupportedError(model_id)

        headers = {}
        if token is not None:
            headers = {"Authorization": f"Bearer {token}"}
        base_url = f"{INFERENCE_ENDPOINT}/models/{model_id}"

        super(APIInferenceAsyncClient, self).__init__(base_url, headers, timeout)


if __name__ == "__main__":
    async def main():
        client = APIInferenceAsyncClient(
            "bigscience/bloomz", token="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"
        )
        async for token in client.generate_stream("test"):
            print(token)

        print(await client.generate("test"))


    import asyncio

    asyncio.run(main())
