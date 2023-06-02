import sys
import subprocess
import contextlib
import pytest
import asyncio
import os
import docker
import json
import math
import time
import random

from docker.errors import NotFound
from typing import Optional, List, Dict
from syrupy.extensions.json import JSONSnapshotExtension
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError

from text_generation import AsyncClient
from text_generation.types import Response, Details, InputToken, Token, BestOfSequence

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", None)
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", None)
DOCKER_VOLUME = os.getenv("DOCKER_VOLUME", "/data")


class ResponseComparator(JSONSnapshotExtension):
    def serialize(
        self,
        data,
        *,
        exclude=None,
        matcher=None,
    ):
        if isinstance(data, List):
            data = [d.dict() for d in data]

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

            if isinstance(data, Dict):
                return Response(**data)
            if isinstance(data, List):
                return [Response(**d) for d in data]
            raise NotImplementedError

        def eq_token(token: Token, other: Token) -> bool:
            return (
                token.id == other.id
                and token.text == other.text
                and math.isclose(token.logprob, other.logprob, rel_tol=0.2)
                and token.special == other.special
            )

        def eq_prefill_token(prefill_token: InputToken, other: InputToken) -> bool:
            try:
                return (
                    prefill_token.id == other.id
                    and prefill_token.text == other.text
                    and (
                        math.isclose(prefill_token.logprob, other.logprob, rel_tol=0.2)
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

        return len(snapshot_data) == len(serialized_data) and all(
            [eq_response(r, o) for r, o in zip(serialized_data, snapshot_data)]
        )


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
            except (ClientConnectorError, ClientOSError, ServerDisconnectedError) as e:
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

        if num_shard is not None:
            args.extend(["--num-shard", str(num_shard)])
        if quantize:
            args.append("--quantize")
            args.append("bitsandbytes")
        if trust_remote_code:
            args.append("--trust-remote-code")

        env = os.environ
        env["LOG_LEVEL"] = "info,text_generation_router=debug"

        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        ) as process:
            yield ProcessLauncherHandle(process, port)

            process.terminate()
            process.wait(60)

            launcher_output = process.stdout.read().decode("utf-8")
            print(launcher_output, file=sys.stderr)

            process.stdout.close()
            process.stderr.close()

    @contextlib.contextmanager
    def docker_launcher(
        model_id: str,
        num_shard: Optional[int] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        port = random.randint(8000, 10_000)

        args = ["--model-id", model_id, "--env"]

        if num_shard is not None:
            args.extend(["--num-shard", str(num_shard)])
        if quantize:
            args.append("--quantize")
            args.append("bitsandbytes")
        if trust_remote_code:
            args.append("--trust-remote-code")

        client = docker.from_env()

        container_name = f"tgi-tests-{model_id.split('/')[-1]}-{num_shard}-{quantize}"

        try:
            container = client.containers.get(container_name)
            container.stop()
            container.wait()
        except NotFound:
            pass

        gpu_count = num_shard if num_shard is not None else 1

        env = {"LOG_LEVEL": "info,text_generation_router=debug"}
        if HUGGING_FACE_HUB_TOKEN is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_HUB_TOKEN

        volumes = []
        if DOCKER_VOLUME:
            volumes = [f"{DOCKER_VOLUME}:/data"]

        container = client.containers.run(
            DOCKER_IMAGE,
            command=args,
            name=container_name,
            environment=env,
            auto_remove=False,
            detach=True,
            device_requests=[
                docker.types.DeviceRequest(count=gpu_count, capabilities=[["gpu"]])
            ],
            volumes=volumes,
            ports={"80/tcp": port},
        )

        yield ContainerLauncherHandle(client, container.name, port)

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
        client: AsyncClient, prompt: str, max_new_tokens: int, n: int
    ) -> List[Response]:
        futures = [
            client.generate(
                prompt, max_new_tokens=max_new_tokens, decoder_input_details=True
            )
            for _ in range(n)
        ]

        return await asyncio.gather(*futures)

    return generate_load_inner
