import asyncio
import contextlib
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from typing import List

import docker
import huggingface_hub
import pytest
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
from docker.errors import NotFound
from huggingface_hub import AsyncInferenceClient, TextGenerationOutput


OPTIMUM_CACHE_REPO_ID = "optimum-internal-testing/neuron-testing-cache"
HF_TOKEN = huggingface_hub.get_token()


def get_tgi_docker_image():
    docker_image = os.getenv("DOCKER_IMAGE", None)
    if docker_image is None:
        client = docker.from_env()
        images = client.images.list(filters={"reference": "text-generation-inference"})
        if not images:
            raise ValueError(
                "No text-generation-inference image found on this host to run tests."
            )
        docker_image = images[0].tags[0]
    return docker_image


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


class TestClient(AsyncInferenceClient):
    def __init__(self, service_name: str, base_url: str):
        super().__init__(model=base_url)
        self.service_name = service_name


class LauncherHandle:
    def __init__(self, service_name: str, port: int):
        self.client = TestClient(service_name, f"http://localhost:{port}")

    def _inner_health(self):
        raise NotImplementedError

    async def health(self, timeout: int = 60):
        assert timeout > 0
        for i in range(timeout):
            if not self._inner_health():
                raise RuntimeError(f"Service crashed after {i} seconds.")

            try:
                await self.client.text_generation("test", max_new_tokens=1)
                logger.info(f"Service started after {i} seconds")
                return
            except (ClientConnectorError, ClientOSError, ServerDisconnectedError):
                time.sleep(1)
            except Exception:
                raise RuntimeError("Basic generation failed with: {e}")
        raise RuntimeError(f"Service failed to start after {i} seconds.")


class ContainerLauncherHandle(LauncherHandle):
    def __init__(self, service_name, docker_client, container_name, port: int):
        super(ContainerLauncherHandle, self).__init__(service_name, port)
        self.docker_client = docker_client
        self.container_name = container_name
        self._log_since = time.time()

    def _inner_health(self) -> bool:
        container = self.docker_client.containers.get(self.container_name)
        container_output = container.logs(since=self._log_since).decode("utf-8")
        self._log_since = time.time()
        if container_output != "":
            print(container_output, end="")
        return container.status in ["running", "created"]


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def neuron_launcher(event_loop):
    """Utility fixture to expose a TGI service.

    The fixture uses a single event loop for each module, but it can create multiple
    docker services with different parameters using the parametrized inner context.

    Args:
        service_name (`str`):
            Used to identify test configurations and adjust test expectations,
        model_name_or_path (`str`):
            The model to use (can be a hub model or a path)
        trust_remote_code (`bool`):
            Must be set to True for gated models.

    Returns:
        A `ContainerLauncherHandle` containing both a TGI server and client.
    """

    @contextlib.contextmanager
    def docker_launcher(
        service_name: str,
        model_name_or_path: str,
        trust_remote_code: bool = False,
    ):
        port = random.randint(8000, 10_000)

        client = docker.from_env()

        container_name = f"tgi-tests-{service_name}-{port}"

        try:
            container = client.containers.get(container_name)
            container.stop()
            container.wait()
        except NotFound:
            pass

        env = {
            "LOG_LEVEL": "info,text_generation_router=debug",
            "CUSTOM_CACHE_REPO": OPTIMUM_CACHE_REPO_ID,
        }

        if HF_TOKEN is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            env["HF_TOKEN"] = HF_TOKEN

        for var in [
            "MAX_BATCH_SIZE",
            "MAX_TOTAL_TOKENS",
            "HF_AUTO_CAST_TYPE",
            "HF_NUM_CORES",
        ]:
            if var in os.environ:
                env[var] = os.environ[var]

        base_image = get_tgi_docker_image()
        if os.path.isdir(model_name_or_path):
            # Create a sub-image containing the model to workaround docker dind issues preventing
            # to share a volume from the container running tests

            test_image = f"{container_name}-img"
            logger.info(
                "Building image on the flight derivated from %s, tagged with %s",
                base_image,
                test_image,
            )
            with tempfile.TemporaryDirectory() as context_dir:
                # Copy model directory to build context
                model_path = os.path.join(context_dir, "model")
                shutil.copytree(model_name_or_path, model_path)
                # Create Dockerfile
                container_model_id = f"/data/{model_name_or_path}"
                docker_content = f"""
                FROM {base_image}
                COPY model {container_model_id}
                """
                with open(os.path.join(context_dir, "Dockerfile"), "wb") as f:
                    f.write(docker_content.encode("utf-8"))
                    f.flush()
                image, logs = client.images.build(
                    path=context_dir, dockerfile=f.name, tag=test_image
                )
            logger.info("Successfully built image %s", image.id)
            logger.debug("Build logs %s", logs)
        else:
            test_image = base_image
            image = None
            container_model_id = model_name_or_path

        args = ["--model-id", container_model_id, "--env"]

        if trust_remote_code:
            args.append("--trust-remote-code")

        container = client.containers.run(
            test_image,
            command=args,
            name=container_name,
            environment=env,
            auto_remove=False,
            detach=True,
            devices=["/dev/neuron0"],
            ports={"80/tcp": port},
            shm_size="1G",
        )

        logger.info(f"Starting {container_name} container")
        yield ContainerLauncherHandle(service_name, client, container.name, port)

        try:
            container.stop(timeout=60)
            container.wait(timeout=60)
        except Exception as e:
            logger.exception(f"Ignoring exception while stopping container: {e}.")
            pass
        finally:
            logger.info("Removing container %s", container_name)
            try:
                container.remove(force=True)
            except Exception as e:
                logger.error(
                    "Error while removing container %s, skipping", container_name
                )
                logger.exception(e)

            # Cleanup the build image
            if image:
                logger.info("Cleaning image %s", image.id)
                try:
                    image.remove(force=True)
                except NotFound:
                    pass
                except Exception as e:
                    logger.error("Error while removing image %s, skipping", image.id)
                    logger.exception(e)

    return docker_launcher


@pytest.fixture(scope="module")
def neuron_generate_load():
    """A utility fixture to launch multiple asynchronous TGI requests in parallel

    Args:
        client (`AsyncClient`):
            An async client
        prompt (`str`):
            The prompt to use (identical for all requests)
        max_new_tokens (`int`):
            The number of tokens to generate for each request.
        n (`int`):
            The number of requests

    Returns:
        A list of `huggingface_hub.TextGenerationOutput`.
    """

    async def generate_load_inner(
        client: AsyncInferenceClient, prompt: str, max_new_tokens: int, n: int
    ) -> List[TextGenerationOutput]:
        futures = [
            client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                details=True,
                decoder_input_details=True,
            )
            for _ in range(n)
        ]

        return await asyncio.gather(*futures)

    return generate_load_inner
