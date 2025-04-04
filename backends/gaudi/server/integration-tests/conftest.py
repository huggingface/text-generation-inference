import asyncio
import contextlib
import os
import shlex
import subprocess
import sys
import threading
import time
from tempfile import TemporaryDirectory
from typing import List
import socket

import docker
import pytest
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
from docker.errors import NotFound
from loguru import logger
from test_model import TEST_CONFIGS
from text_generation import AsyncClient
from text_generation.types import Response

# Use the latest image from the local docker build
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "tgi-gaudi")
DOCKER_VOLUME = os.getenv("DOCKER_VOLUME", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)

assert (
    HF_TOKEN is not None
), "HF_TOKEN is not set, please set it as some models are gated and thus the test will fail without it"

if DOCKER_VOLUME is None:
    logger.warning(
        "DOCKER_VOLUME is not set, this will lead to the tests redownloading the models on each run, consider setting it to speed up testing"
    )

LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

BASE_ENV = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "LOG_LEVEL": LOG_LEVEL,
    "HF_TOKEN": os.getenv("HF_TOKEN", None),
}


HABANA_RUN_ARGS = {
    "runtime": "habana",
    "ipc_mode": "host",
    "cap_add": ["sys_nice"],
}

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def stream_container_logs(container, test_name):
    """Stream container logs in a separate thread."""
    try:
        for log in container.logs(stream=True, follow=True):
            print(
                f"[TGI Server Logs - {test_name}] {log.decode('utf-8')}",
                end="",
                file=sys.stderr,
                flush=True,
            )
    except Exception as e:
        logger.error(f"Error streaming container logs: {str(e)}")


class LauncherHandle:
    def __init__(self, port: int):
        self.client = AsyncClient(f"http://localhost:{port}", timeout=3600)

    def _inner_health(self):
        raise NotImplementedError

    async def health(self, timeout: int = 60):
        assert timeout > 0
        start_time = time.time()
        logger.info(f"Starting health check with timeout of {timeout}s")

        for attempt in range(timeout):
            if not self._inner_health():
                logger.error("Launcher crashed during health check")
                raise RuntimeError("Launcher crashed")

            try:
                await self.client.generate("test")
                elapsed = time.time() - start_time
                logger.info(f"Health check passed after {elapsed:.1f}s")
                return
            except (ClientConnectorError, ClientOSError, ServerDisconnectedError) as e:
                if attempt == timeout - 1:
                    logger.error(f"Health check failed after {timeout}s: {str(e)}")
                    raise RuntimeError(f"Health check failed: {str(e)}")
                if attempt % 10 == 0 and attempt != 0:  # Only log every 10th attempt
                    logger.debug(
                        f"Connection attempt {attempt}/{timeout} failed: {str(e)}"
                    )
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error during health check: {str(e)}")
                # Get full traceback for debugging
                import traceback

                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise


class ContainerLauncherHandle(LauncherHandle):
    def __init__(self, docker_client, container_name, port: int):
        super(ContainerLauncherHandle, self).__init__(port)
        self.docker_client = docker_client
        self.container_name = container_name

    def _inner_health(self) -> bool:
        try:
            container = self.docker_client.containers.get(self.container_name)
            status = container.status
            if status not in ["running", "created"]:
                logger.warning(f"Container status is {status}")
                # Get container logs for debugging
                logs = container.logs().decode("utf-8")
                logger.debug(f"Container logs:\n{logs}")
            return status in ["running", "created"]
        except Exception as e:
            logger.error(f"Error checking container health: {str(e)}")
            return False


class ProcessLauncherHandle(LauncherHandle):
    def __init__(self, process, port: int):
        super(ProcessLauncherHandle, self).__init__(port)
        self.process = process

    def _inner_health(self) -> bool:
        return self.process.poll() is None


@pytest.fixture(scope="module")
def data_volume():
    tmpdir = TemporaryDirectory()
    yield tmpdir.name
    try:
        # Cleanup the temporary directory using sudo as it contains root files created by the container
        subprocess.run(shlex.split(f"sudo rm -rf {tmpdir.name}"), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")


@pytest.fixture(scope="module")
def launcher(data_volume):
    @contextlib.contextmanager
    def docker_launcher(
        model_id: str,
        test_name: str,
    ):
        logger.info(
            f"Starting docker launcher for model {model_id} and test {test_name}"
        )

        # Get a random available port
        def get_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port

        port = get_free_port()
        logger.debug(f"Using port {port}")

        client = docker.from_env()

        container_name = f"tgi-gaudi-test-{test_name.replace('/', '-')}"

        try:
            container = client.containers.get(container_name)
            logger.info(
                f"Stopping existing container {container_name} for test {test_name}"
            )
            container.stop()
            container.wait()
        except NotFound:
            pass
        except Exception as e:
            logger.error(f"Error handling existing container: {str(e)}")

        model_name = next(
            name for name, cfg in TEST_CONFIGS.items() if cfg["model_id"] == model_id
        )

        tgi_args = TEST_CONFIGS[model_name]["args"].copy()

        env = BASE_ENV.copy()

        # Add model_id to env
        env["MODEL_ID"] = model_id

        # Add env config that is definied in the fixture parameter
        if "env_config" in TEST_CONFIGS[model_name]:
            env.update(TEST_CONFIGS[model_name]["env_config"].copy())

        volumes = [f"{DOCKER_VOLUME}:/data"]
        logger.debug(f"Using volume {volumes}")

        try:
            logger.info(f"Creating container with name {container_name}")

            # Log equivalent docker run command for debugging, this is not actually executed
            container = client.containers.run(
                DOCKER_IMAGE,
                command=tgi_args,
                name=container_name,
                environment=env,
                detach=True,
                volumes=volumes,
                ports={"80/tcp": port},
                **HABANA_RUN_ARGS,
            )

            logger.info(f"Container {container_name} started successfully")

            # Start log streaming in a background thread
            log_thread = threading.Thread(
                target=stream_container_logs,
                args=(container, test_name),
                daemon=True,  # This ensures the thread will be killed when the main program exits
            )
            log_thread.start()

            # Add a small delay to allow container to initialize
            time.sleep(2)

            # Check container status after creation
            status = container.status
            logger.debug(f"Initial container status: {status}")
            if status not in ["running", "created"]:
                logs = container.logs().decode("utf-8")
                logger.error(f"Container failed to start properly. Logs:\n{logs}")

            yield ContainerLauncherHandle(client, container.name, port)

        except Exception as e:
            logger.error(f"Error starting container: {str(e)}")
            # Get full traceback for debugging
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        finally:
            try:
                container = client.containers.get(container_name)
                logger.info(f"Stopping container {container_name}")
                container.stop()
                container.wait()

                container_output = container.logs().decode("utf-8")
                print(container_output, file=sys.stderr)

                container.remove()
                logger.info(f"Container {container_name} removed successfully")
            except NotFound:
                pass
            except Exception as e:
                logger.warning(f"Error cleaning up container: {str(e)}")

    return docker_launcher


@pytest.fixture(scope="module")
def generate_load():
    async def generate_load_inner(
        client: AsyncClient, prompt: str, max_new_tokens: int, n: int
    ) -> List[Response]:
        try:
            futures = [
                client.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    decoder_input_details=True,
                )
                for _ in range(n)
            ]
            return await asyncio.gather(*futures)
        except Exception as e:
            logger.error(f"Error generating load: {str(e)}")
            raise

    return generate_load_inner
