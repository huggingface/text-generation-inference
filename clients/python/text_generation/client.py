import json
import requests

from aiohttp import ClientSession, ClientTimeout
from pydantic import ValidationError
from typing import Dict, Optional, List, AsyncIterator, Iterator

from text_generation.types import (
    StreamResponse,
    Response,
    Request,
    Parameters,
)
from text_generation.errors import parse_error


class Client:
    def __init__(
        self, base_url: str, headers: Dict[str, str] = None, timeout: int = 10
    ):
        self.base_url = base_url
        self.headers = headers
        self.timeout = timeout

    def generate(
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

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return Response(**payload[0])

    def generate_stream(
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
    ) -> Iterator[StreamResponse]:
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

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            timeout=self.timeout,
            stream=True,
        )

        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json())

        for byte_payload in resp.iter_lines():
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            if payload.startswith("data:"):
                json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                try:
                    response = StreamResponse(**json_payload)
                except ValidationError:
                    raise parse_error(resp.status_code, json_payload)
                yield response


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
