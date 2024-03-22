import json
import requests

from aiohttp import ClientSession, ClientTimeout
from pydantic import ValidationError
from typing import Dict, Optional, List, AsyncIterator, Iterator, Union

from text_generation.types import (
    StreamResponse,
    Response,
    Request,
    Parameters,
    Grammar,
    ChatRequest,
    ChatCompletionChunk,
    ChatComplete,
    Message,
    Tool,
)
from text_generation.errors import parse_error


class Client:
    """Client to make calls to a text-generation-inference instance

     Example:

     ```python
     >>> from text_generation import Client

     >>> client = Client("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> client.generate("Why is the sky blue?").generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> for response in client.generate_stream("Why is the sky blue?"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-generation-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout

    def chat(
        self,
        messages: List[Message],
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
    ):
        """
        Given a list of messages, generate a response asynchronously

        Args:
            messages (`List[Message]`):
                List of messages
            repetition_penalty (`float`):
                The parameter for repetition penalty. 0.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            frequency_penalty (`float`):
                The parameter for frequency penalty. 0.0 means no penalty
                Penalize new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            logit_bias (`List[float]`):
                Adjust the likelihood of specified tokens
            logprobs (`bool`):
                Include log probabilities in the response
            top_logprobs (`int`):
                Include the `n` most likely tokens at each step
            max_tokens (`int`):
                Maximum number of generated tokens
            n (`int`):
                Generate `n` completions
            presence_penalty (`float`):
                The parameter for presence penalty. 0.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            stream (`bool`):
                Stream the response
            seed (`int`):
                Random sampling seed
            temperature (`float`):
                The value used to module the logits distribution.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation
            tools (`List[Tool]`):
                List of tools to use
            tool_choice (`str`):
                The tool to use

        """
        request = ChatRequest(
            model="tgi",
            messages=messages,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            stream=stream,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
        )
        if not stream:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=request.dict(),
                headers=self.headers,
                cookies=self.cookies,
                timeout=self.timeout,
            )
            payload = resp.json()
            if resp.status_code != 200:
                raise parse_error(resp.status_code, payload)
            return ChatComplete(**payload)
        else:
            return self._chat_stream_response(request)

    def _chat_stream_response(self, request):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
            stream=True,
        )
        # iterate and print stream
        for byte_payload in resp.iter_lines():
            if byte_payload == b"\n":
                continue
            payload = byte_payload.decode("utf-8")
            if payload.startswith("data:"):
                json_payload = json.loads(payload.lstrip("data:").rstrip("\n"))
                try:
                    response = ChatCompletionChunk(**json_payload)
                    yield response
                except ValidationError:
                    raise parse_error(resp.status, json_payload)

    def generate(
        self,
        prompt: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> Response:
        """
        Given a prompt, generate the following text

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            frequency_penalty (`float`):
                The parameter for frequency penalty. 1.0 means no penalty
                Penalize new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step
            grammar (`Grammar`):
                Whether to use a grammar for the generation and the grammar to use. Grammars will constrain the generation
                of the text to match a regular expression or JSON schema.

        Returns:
            Response: generated response
        """
        # Validate parameters
        parameters = Parameters(
            best_of=best_of,
            details=True,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            decoder_input_details=decoder_input_details,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
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
        frequency_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> Iterator[StreamResponse]:
        """
        Given a prompt, generate the following stream of tokens

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            frequency_penalty (`float`):
                The parameter for frequency penalty. 1.0 means no penalty
                Penalize new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step
            grammar (`Grammar`):
                Whether to use a grammar for the generation and the grammar to use. Grammars will constrain the generation
                of the text to match a regular expression or JSON schema.

        Returns:
            Iterator[StreamResponse]: stream of generated tokens
        """
        # Validate parameters
        parameters = Parameters(
            best_of=None,
            details=True,
            decoder_input_details=False,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=True, parameters=parameters)

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
            stream=True,
        )

        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json())

        # Parse ServerSentEvents
        for byte_payload in resp.iter_lines():
            # Skip line
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            # Event data
            if payload.startswith("data:"):
                # Decode payload
                json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                # Parse payload
                try:
                    response = StreamResponse(**json_payload)
                except ValidationError:
                    # If we failed to parse the payload, then it is an error payload
                    raise parse_error(resp.status_code, json_payload)
                yield response


class AsyncClient:
    """Asynchronous Client to make calls to a text-generation-inference instance

     Example:

     ```python
     >>> from text_generation import AsyncClient

     >>> client = AsyncClient("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> response = await client.generate("Why is the sky blue?")
     >>> response.generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> async for response in client.generate_stream("Why is the sky blue?"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-generation-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = ClientTimeout(timeout)

    async def chat(
        self,
        messages: List[Message],
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
    ) -> Union[ChatComplete, AsyncIterator[ChatCompletionChunk]]:
        """
        Given a list of messages, generate a response asynchronously

        Args:
            messages (`List[Message]`):
                List of messages
            repetition_penalty (`float`):
                The parameter for frequency penalty. 0.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            frequency_penalty (`float`):
                The parameter for frequency penalty. 0.0 means no penalty
                Penalize new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            logit_bias (`List[float]`):
                Adjust the likelihood of specified tokens
            logprobs (`bool`):
                Include log probabilities in the response
            top_logprobs (`int`):
                Include the `n` most likely tokens at each step
            max_tokens (`int`):
                Maximum number of generated tokens
            n (`int`):
                Generate `n` completions
            presence_penalty (`float`):
                The parameter for presence penalty. 0.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            stream (`bool`):
                Stream the response
            seed (`int`):
                Random sampling seed
            temperature (`float`):
                The value used to module the logits distribution.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation
            tools (`List[Tool]`):
                List of tools to use
            tool_choice (`str`):
                The tool to use

        """
        request = ChatRequest(
            model="tgi",
            messages=messages,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            stream=stream,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
        )
        if not stream:
            return await self._chat_single_response(request)
        else:
            return self._chat_stream_response(request)

    async def _chat_single_response(self, request):
        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions", json=request.dict()
            ) as resp:
                payload = await resp.json()
                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return ChatComplete(**payload)

    async def _chat_stream_response(self, request):
        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions", json=request.dict()
            ) as resp:
                async for byte_payload in resp.content:
                    if byte_payload == b"\n":
                        continue
                    payload = byte_payload.decode("utf-8")
                    if payload.startswith("data:"):
                        json_payload = json.loads(payload.lstrip("data:").rstrip("\n"))
                        try:
                            response = ChatCompletionChunk(**json_payload)
                            yield response
                        except ValidationError:
                            raise parse_error(resp.status, json_payload)

    async def generate(
        self,
        prompt: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> Response:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            frequency_penalty (`float`):
                The parameter for frequency penalty. 1.0 means no penalty
                Penalize new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step
            grammar (`Grammar`):
                Whether to use a grammar for the generation and the grammar to use. Grammars will constrain the generation
                of the text to match a regular expression or JSON schema.

        Returns:
            Response: generated response
        """

        # Validate parameters
        parameters = Parameters(
            best_of=best_of,
            details=True,
            decoder_input_details=decoder_input_details,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
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
        frequency_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> AsyncIterator[StreamResponse]:
        """
        Given a prompt, generate the following stream of tokens asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            frequency_penalty (`float`):
                The parameter for frequency penalty. 1.0 means no penalty
                Penalize new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step
            grammar (`Grammar`):
                Whether to use a grammar for the generation and the grammar to use. Grammars will constrain the generation
                of the text to match a regular expression or JSON schema.

        Returns:
            AsyncIterator[StreamResponse]: stream of generated tokens
        """
        # Validate parameters
        parameters = Parameters(
            best_of=None,
            details=True,
            decoder_input_details=False,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=True, parameters=parameters)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.dict()) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json())

                # Parse ServerSentEvents
                async for byte_payload in resp.content:
                    # Skip line
                    if byte_payload == b"\n":
                        continue

                    payload = byte_payload.decode("utf-8")

                    # Event data
                    if payload.startswith("data:"):
                        # Decode payload
                        json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                        # Parse payload
                        try:
                            response = StreamResponse(**json_payload)
                        except ValidationError:
                            # If we failed to parse the payload, then it is an error payload
                            raise parse_error(resp.status, json_payload)
                        yield response
