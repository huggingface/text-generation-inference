import subprocess
import time
import contextlib
import pytest
import asyncio
import os

from typing import Optional, List
from aiohttp import ClientConnectorError

from text_generation import AsyncClient
from text_generation.types import Response

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", None)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def launcher(event_loop):
    @contextlib.asynccontextmanager
    async def local_launcher_inner(model_id: str, num_shard: Optional[int] = None, quantize: bool = False):
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
            client = AsyncClient(f"http://localhost:{port}")

            healthy = False

            for _ in range(60):
                launcher_output = process.stdout.read1().decode("utf-8")
                print(launcher_output)

                exit_code = process.poll()
                if exit_code is not None:
                    launcher_error = process.stderr.read1().decode("utf-8")
                    print(launcher_error)
                    raise RuntimeError(
                        f"text-generation-launcher terminated with exit code {exit_code}"
                    )

                try:
                    await client.generate("test", max_new_tokens=1)
                    healthy = True
                    break
                except ClientConnectorError:
                    time.sleep(1)

            if healthy:
                yield client

            process.terminate()

            for _ in range(60):
                exit_code = process.wait(1)
                if exit_code is not None:
                    break

            launcher_output = process.stdout.read1().decode("utf-8")
            print(launcher_output)

            process.stdout.close()
            process.stderr.close()

            if not healthy:
                raise RuntimeError(f"unable to start model {model_id} with command: {' '.join(args)}")

    return launcher_inner


@pytest.fixture(scope="module")
def generate_load():
    async def generate_load_inner(client: AsyncClient, prompt: str, max_new_tokens: int, n: int) -> List[Response]:
        futures = [client.generate(prompt, max_new_tokens=max_new_tokens) for _ in range(n)]

        results = await asyncio.gather(*futures)
        return results

    return generate_load_inner
