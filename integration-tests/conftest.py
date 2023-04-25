import pytest
import subprocess
import time
import contextlib

from text_generation import Client
from typing import Optional
from requests import ConnectionError


@contextlib.contextmanager
def launcher(model_id: str, num_shard: Optional[int] = None, quantize: bool = False):
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
        args.extend(["--num-shard", num_shard])
    if quantize:
        args.append("--quantize")

    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        client = Client(f"http://localhost:{port}")

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
                client.generate("test", max_new_tokens=1)
                break
            except ConnectionError:
                time.sleep(1)

        yield client

        process.stdout.close()
        process.stderr.close()
        process.terminate()


@pytest.fixture(scope="session")
def bloom_560m():
    with launcher("bigscience/bloom-560m") as client:
        yield client


@pytest.fixture(scope="session")
def bloom_560m_multi():
    with launcher("bigscience/bloom-560m", num_shard=2) as client:
        yield client
