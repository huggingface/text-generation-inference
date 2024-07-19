import asyncio
import contextlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import docker
import pytest
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
from docker.errors import NotFound
from syrupy.extensions.json import JSONSnapshotExtension
from text_generation import AsyncClient
from text_generation.types import (
    BestOfSequence,
    ChatComplete,
    ChatCompletionChunk,
    ChatCompletionComplete,
    Completion,
    Details,
    Grammar,
    InputToken,
    Response,
    Token,
)

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)
DOCKER_VOLUME = os.getenv("DOCKER_VOLUME", "/data")
DOCKER_DEVICES = os.getenv("DOCKER_DEVICES")


def pytest_addoption(parser):
    parser.addoption(
        "--release", action="store_true", default=False, help="run release tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "release: mark test as a release-only test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--release"):
        # --release given in cli: do not skip release tests
        return
    skip_release = pytest.mark.skip(reason="need --release option to run")
    for item in items:
        if "release" in item.keywords:
            item.add_marker(skip_release)


class ResponseComparator(JSONSnapshotExtension):
    rtol = 0.2
    ignore_logprob = False

    def serialize(
        self,
        data,
        *,
        exclude=None,
        matcher=None,
    ):
        if (
            isinstance(data, Response)
            or isinstance(data, ChatComplete)
            or isinstance(data, ChatCompletionChunk)
            or isinstance(data, ChatCompletionComplete)
        ):
            data = data.model_dump()

        if isinstance(data, List):
            data = [d.model_dump() for d in data]

        data = self._filter(
            data=data, depth=0, path=(), exclude=exclude, matcher=matcher
        )
        return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"

    def matches(
        self,
        *,
        serialized_data,
        snapshot_data,
    ) -> bool:
        def convert_data(data):
            data = json.loads(data)
            if isinstance(data, Dict) and "choices" in data:
                choices = data["choices"]
                if isinstance(choices, List) and len(choices) >= 1:
                    if "delta" in choices[0]:
                        return ChatCompletionChunk(**data)
                    if "text" in choices[0]:
                        return Completion(**data)
                return ChatComplete(**data)

            if isinstance(data, Dict):
                return Response(**data)
            if isinstance(data, List):
                if (
                    len(data) > 0
                    and "object" in data[0]
                    and data[0]["object"] == "text_completion"
                ):
                    return [Completion(**d) for d in data]
                return [Response(**d) for d in data]
            raise NotImplementedError

        def eq_token(token: Token, other: Token) -> bool:
            return (
                token.id == other.id
                and token.text == other.text
                and (
                    self.ignore_logprob
                    or math.isclose(token.logprob, other.logprob, rel_tol=self.rtol)
                )
                and token.special == other.special
            )

        def eq_prefill_token(prefill_token: InputToken, other: InputToken) -> bool:
            try:
                return (
                    prefill_token.id == other.id
                    and prefill_token.text == other.text
                    and (
                        self.ignore_logprob
                        or math.isclose(
                            prefill_token.logprob,
                            other.logprob,
                            rel_tol=self.rtol,
                        )
                        if prefill_token.logprob is not None
                        else prefill_token.logprob == other.logprob
                    )
                )
            except TypeError:
                return False

        def eq_best_of(details: BestOfSequence, other: BestOfSequence) -> bool:
            return (
                details.finish_reason == other.finish_reason
                and details.generated_tokens == other.generated_tokens
                and details.seed == other.seed
                and len(details.prefill) == len(other.prefill)
                and all(
                    [
                        eq_prefill_token(d, o)
                        for d, o in zip(details.prefill, other.prefill)
                    ]
                )
                and len(details.tokens) == len(other.tokens)
                and all([eq_token(d, o) for d, o in zip(details.tokens, other.tokens)])
            )

        def eq_details(details: Details, other: Details) -> bool:
            return (
                details.finish_reason == other.finish_reason
                and details.generated_tokens == other.generated_tokens
                and details.seed == other.seed
                and len(details.prefill) == len(other.prefill)
                and all(
                    [
                        eq_prefill_token(d, o)
                        for d, o in zip(details.prefill, other.prefill)
                    ]
                )
                and len(details.tokens) == len(other.tokens)
                and all([eq_token(d, o) for d, o in zip(details.tokens, other.tokens)])
                and (
                    len(details.best_of_sequences)
                    if details.best_of_sequences is not None
                    else 0
                )
                == (
                    len(other.best_of_sequences)
                    if other.best_of_sequences is not None
                    else 0
                )
                and (
                    all(
                        [
                            eq_best_of(d, o)
                            for d, o in zip(
                                details.best_of_sequences, other.best_of_sequences
                            )
                        ]
                    )
                    if details.best_of_sequences is not None
                    else details.best_of_sequences == other.best_of_sequences
                )
            )

        def eq_completion(response: Completion, other: Completion) -> bool:
            return response.choices[0].text == other.choices[0].text

        def eq_chat_complete(response: ChatComplete, other: ChatComplete) -> bool:
            return (
                response.choices[0].message.content == other.choices[0].message.content
            )

        def eq_chat_complete_chunk(
            response: ChatCompletionChunk, other: ChatCompletionChunk
        ) -> bool:
            return response.choices[0].delta.content == other.choices[0].delta.content

        def eq_response(response: Response, other: Response) -> bool:
            return response.generated_text == other.generated_text and eq_details(
                response.details, other.details
            )

        serialized_data = convert_data(serialized_data)
        snapshot_data = convert_data(snapshot_data)

        if not isinstance(serialized_data, List):
            serialized_data = [serialized_data]
        if not isinstance(snapshot_data, List):
            snapshot_data = [snapshot_data]

        if isinstance(serialized_data[0], Completion):
            return len(snapshot_data) == len(serialized_data) and all(
                [eq_completion(r, o) for r, o in zip(serialized_data, snapshot_data)]
            )

        if isinstance(serialized_data[0], ChatComplete):
            return len(snapshot_data) == len(serialized_data) and all(
                [eq_chat_complete(r, o) for r, o in zip(serialized_data, snapshot_data)]
            )

        if isinstance(serialized_data[0], ChatCompletionChunk):
            return len(snapshot_data) == len(serialized_data) and all(
                [
                    eq_chat_complete_chunk(r, o)
                    for r, o in zip(serialized_data, snapshot_data)
                ]
            )

        return len(snapshot_data) == len(serialized_data) and all(
            [eq_response(r, o) for r, o in zip(serialized_data, snapshot_data)]
        )


class GenerousResponseComparator(ResponseComparator):
    # Needed for GPTQ with exllama which has serious numerical fluctuations.
    rtol = 0.75


class IgnoreLogProbResponseComparator(ResponseComparator):
    ignore_logprob = True


class LauncherHandle:
    def __init__(self, port: int):
        self.client = AsyncClient(f"http://localhost:{port}")

    def _inner_health(self):
        raise NotImplementedError

    async def health(self, timeout: int = 60):
        assert timeout > 0
        for _ in range(timeout):
            if not self._inner_health():
                raise RuntimeError("Launcher crashed")

            try:
                await self.client.generate("test")
                return
            except (ClientConnectorError, ClientOSError, ServerDisconnectedError):
                time.sleep(1)
        raise RuntimeError("Health check failed")


class ContainerLauncherHandle(LauncherHandle):
    def __init__(self, docker_client, container_name, port: int):
        super(ContainerLauncherHandle, self).__init__(port)
        self.docker_client = docker_client
        self.container_name = container_name

    def _inner_health(self) -> bool:
        container = self.docker_client.containers.get(self.container_name)
        return container.status in ["running", "created"]


class ProcessLauncherHandle(LauncherHandle):
    def __init__(self, process, port: int):
        super(ProcessLauncherHandle, self).__init__(port)
        self.process = process

    def _inner_health(self) -> bool:
        return self.process.poll() is None


@pytest.fixture
def response_snapshot(snapshot):
    return snapshot.use_extension(ResponseComparator)


@pytest.fixture
def generous_response_snapshot(snapshot):
    return snapshot.use_extension(GenerousResponseComparator)


@pytest.fixture
def ignore_logprob_response_snapshot(snapshot):
    return snapshot.use_extension(IgnoreLogProbResponseComparator)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def launcher(event_loop):
    @contextlib.contextmanager
    def local_launcher(
        model_id: str,
        num_shard: Optional[int] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
        use_flash_attention: bool = True,
        disable_grammar_support: bool = False,
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        max_input_length: Optional[int] = None,
        max_batch_prefill_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        lora_adapters: Optional[List[str]] = None,
        cuda_graphs: Optional[List[int]] = None,
    ):
        port = random.randint(8000, 10_000)
        master_port = random.randint(10_000, 20_000)

        shard_uds_path = (
            f"/tmp/tgi-tests-{model_id.split('/')[-1]}-{num_shard}-{quantize}-server"
        )

        args = [
            "text-generation-launcher",
            "--model-id",
            model_id,
            "--port",
            str(port),
            "--master-port",
            str(master_port),
            "--shard-uds-path",
            shard_uds_path,
        ]

        env = os.environ

        if disable_grammar_support:
            args.append("--disable-grammar-support")
        if num_shard is not None:
            args.extend(["--num-shard", str(num_shard)])
        if quantize is not None:
            args.append("--quantize")
            args.append(quantize)
        if dtype is not None:
            args.append("--dtype")
            args.append(dtype)
        if revision is not None:
            args.append("--revision")
            args.append(revision)
        if trust_remote_code:
            args.append("--trust-remote-code")
        if max_input_length:
            args.append("--max-input-length")
            args.append(str(max_input_length))
        if max_batch_prefill_tokens:
            args.append("--max-batch-prefill-tokens")
            args.append(str(max_batch_prefill_tokens))
        if max_total_tokens:
            args.append("--max-total-tokens")
            args.append(str(max_total_tokens))
        if lora_adapters:
            args.append("--lora-adapters")
            args.append(",".join(lora_adapters))
        if cuda_graphs:
            args.append("--cuda-graphs")
            args.append(",".join(map(str, cuda_graphs)))

        print(" ".join(args), file=sys.stderr)

        env["LOG_LEVEL"] = "info,text_generation_router=debug"

        if not use_flash_attention:
            env["USE_FLASH_ATTENTION"] = "false"

        with tempfile.TemporaryFile("w+") as tmp:
            # We'll output stdout/stderr to a temporary file. Using a pipe
            # cause the process to block until stdout is read.
            with subprocess.Popen(
                args,
                stdout=tmp,
                stderr=subprocess.STDOUT,
                env=env,
            ) as process:
                yield ProcessLauncherHandle(process, port)

                process.terminate()
                process.wait(60)

                tmp.seek(0)
                shutil.copyfileobj(tmp, sys.stderr)

        if not use_flash_attention:
            del env["USE_FLASH_ATTENTION"]

    @contextlib.contextmanager
    def docker_launcher(
        model_id: str,
        num_shard: Optional[int] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
        use_flash_attention: bool = True,
        disable_grammar_support: bool = False,
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        max_input_length: Optional[int] = None,
        max_batch_prefill_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        lora_adapters: Optional[List[str]] = None,
        cuda_graphs: Optional[List[int]] = None,
    ):
        port = random.randint(8000, 10_000)

        args = ["--model-id", model_id, "--env"]

        if disable_grammar_support:
            args.append("--disable-grammar-support")
        if num_shard is not None:
            args.extend(["--num-shard", str(num_shard)])
        if quantize is not None:
            args.append("--quantize")
            args.append(quantize)
        if dtype is not None:
            args.append("--dtype")
            args.append(dtype)
        if revision is not None:
            args.append("--revision")
            args.append(revision)
        if trust_remote_code:
            args.append("--trust-remote-code")
        if max_input_length:
            args.append("--max-input-length")
            args.append(str(max_input_length))
        if max_batch_prefill_tokens:
            args.append("--max-batch-prefill-tokens")
            args.append(str(max_batch_prefill_tokens))
        if max_total_tokens:
            args.append("--max-total-tokens")
            args.append(str(max_total_tokens))
        if lora_adapters:
            args.append("--lora-adapters")
            args.append(",".join(lora_adapters))
        if cuda_graphs:
            args.append("--cuda-graphs")
            args.append(",".join(map(str, cuda_graphs)))

        client = docker.from_env()

        container_name = f"tgi-tests-{model_id.split('/')[-1]}-{num_shard}-{quantize}"

        try:
            container = client.containers.get(container_name)
            container.stop()
            container.wait()
        except NotFound:
            pass

        gpu_count = num_shard if num_shard is not None else 1

        env = {
            "LOG_LEVEL": "info,text_generation_router=debug",
        }
        if not use_flash_attention:
            env["USE_FLASH_ATTENTION"] = "false"

        if HF_TOKEN is not None:
            env["HF_TOKEN"] = HF_TOKEN

        volumes = []
        if DOCKER_VOLUME:
            volumes = [f"{DOCKER_VOLUME}:/data"]

        if DOCKER_DEVICES:
            devices = DOCKER_DEVICES.split(",")
            visible = os.getenv("ROCR_VISIBLE_DEVICES")
            if visible:
                env["ROCR_VISIBLE_DEVICES"] = visible
            device_requests = []
        else:
            devices = []
            device_requests = [
                docker.types.DeviceRequest(count=gpu_count, capabilities=[["gpu"]])
            ]

        container = client.containers.run(
            DOCKER_IMAGE,
            command=args,
            name=container_name,
            environment=env,
            auto_remove=False,
            detach=True,
            device_requests=device_requests,
            devices=devices,
            volumes=volumes,
            ports={"80/tcp": port},
            shm_size="1G",
        )

        yield ContainerLauncherHandle(client, container.name, port)

        if not use_flash_attention:
            del env["USE_FLASH_ATTENTION"]

        try:
            container.stop()
            container.wait()
        except NotFound:
            pass

        container_output = container.logs().decode("utf-8")
        print(container_output, file=sys.stderr)

        container.remove()

    if DOCKER_IMAGE is not None:
        return docker_launcher
    return local_launcher


@pytest.fixture(scope="module")
def generate_load():
    async def generate_load_inner(
        client: AsyncClient,
        prompt: str,
        max_new_tokens: int,
        n: int,
        seed: Optional[int] = None,
        grammar: Optional[Grammar] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[Response]:
        futures = [
            client.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                decoder_input_details=True,
                seed=seed,
                grammar=grammar,
                stop_sequences=stop_sequences,
            )
            for _ in range(n)
        ]

        return await asyncio.gather(*futures)

    return generate_load_inner
