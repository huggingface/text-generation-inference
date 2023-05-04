import subprocess
import contextlib
import pytest
import asyncio
import os
import docker

from docker.errors import NotFound
from typing import Optional, List

from text_generation import AsyncClient
from text_generation.types import Response

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", None)
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", None)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def launcher(event_loop):
    @contextlib.contextmanager
    def local_launcher(
        model_id: str, num_shard: Optional[int] = None, quantize: bool = False
    ):
        port = 9999
        master_port = 19999

        shard_uds_path = f"/tmp/{model_id.replace('/', '--')}-server"

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

        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            yield AsyncClient(f"http://localhost:{port}")

            process.terminate()
            process.wait(60)

            launcher_output = process.stdout.read1().decode("utf-8")
            print(launcher_output)

            process.stdout.close()
            process.stderr.close()

    @contextlib.contextmanager
    def docker_launcher(
        model_id: str, num_shard: Optional[int] = None, quantize: bool = False
    ):
        port = 9999

        args = ["--model-id", model_id, "--env"]

        if num_shard is not None:
            args.extend(["--num-shard", str(num_shard)])
        if quantize:
            args.append("--quantize")

        client = docker.from_env()

        container_name = f"tgi-tests-{model_id.split('/')[-1]}-{num_shard}-{quantize}"

        try:
            container = client.containers.get(container_name)
            container.stop()
            container.wait()
        except NotFound:
            pass

        gpu_count = num_shard if num_shard is not None else 1

        env = {}
        if HUGGING_FACE_HUB_TOKEN is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_HUB_TOKEN

        container = client.containers.run(
            DOCKER_IMAGE,
            command=args,
            name=container_name,
            environment=env,
            auto_remove=True,
            detach=True,
            device_requests=[
                docker.types.DeviceRequest(count=gpu_count, capabilities=[["gpu"]])
            ],
            volumes=["/data:/data"],
            ports={"80/tcp": port},
        )

        yield AsyncClient(f"http://localhost:{port}")

        container.stop()

        container_output = container.logs().decode("utf-8")
        print(container_output)

    if DOCKER_IMAGE is not None:
        return docker_launcher
    return local_launcher


@pytest.fixture(scope="module")
def generate_load():
    async def generate_load_inner(
        client: AsyncClient, prompt: str, max_new_tokens: int, n: int
    ) -> List[Response]:
        futures = [
            client.generate(prompt, max_new_tokens=max_new_tokens) for _ in range(n)
        ]

        results = await asyncio.gather(*futures)
        return results

    return generate_load_inner
