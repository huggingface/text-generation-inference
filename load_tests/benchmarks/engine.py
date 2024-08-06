import subprocess
import threading
from typing import Dict, List

import docker
from docker.models.containers import Container
from loguru import logger

from benchmarks.utils import kill


class InferenceEngineRunner:
    def __init__(self, model: str):
        self.model = model

    def run(self, parameters: list[tuple]):
        NotImplementedError("This method should be implemented by the subclass")

    def stop(self):
        NotImplementedError("This method should be implemented by the subclass")


class TGIRunner(InferenceEngineRunner):
    def __init__(self, model: str):
        super().__init__(model)
        self.process = None
        self.model = model

    def run(self, parameters: list[tuple]):
        params = ""
        for p in parameters:
            params += f"--{p[0]} {str(p[1])}"
        # start a TGI subprocess with the given parameter
        args = f"text-generation-launcher --port 8080 --model-id {self.model} --huggingface-hub-cache /scratch {params}"
        logger.info(f"Running TGI with parameters: {args}")
        self.process = subprocess.Popen(args,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        for line in iter(self.process.stdout.readline, b""):
            print(line.decode("utf-8"))
            # wait for TGI to listen to the port
            if b"Connected" in line:
                break
            if b"Error" in line:
                raise Exception(f"Error starting TGI: {line}")

        # continue to stream the logs in a thread
        def stream_logs():
            for line in iter(self.process.stdout.readline, b""):
                print(line.decode("utf-8"))

        if self.process.returncode is not None:
            raise Exception("Error starting TGI")
        self.thread = threading.Thread(target=stream_logs)
        self.thread.start()

    def stop(self):
        logger.warning(f"Killing TGI with PID {self.process.pid}")
        if self.process:
            kill(self.process.pid)
        if self.thread:
            self.thread.join()


class TGIDockerRunner(InferenceEngineRunner):
    def __init__(self,
                 model: str,
                 image: str = "ghcr.io/huggingface/text-generation-inference:latest",
                 volumes=None):
        super().__init__(model)
        if volumes is None:
            volumes = []
        self.container = None
        self.image = image
        self.volumes = volumes

    def run(self, parameters: list[tuple]):
        params = f"--model-id {self.model} --port 8080"
        for p in parameters:
            params += f" --{p[0]} {str(p[1])}"
        logger.info(f"Running TGI with parameters: {params}")
        volumes = {}
        for v in self.volumes:
            volumes[v[0]] = {"bind": v[1], "mode": "rw"}
        self.container = run_docker(self.image, params,
                                    "Connected",
                                    "Error",
                                    volumes=volumes)

    def stop(self):
        if self.container:
            self.container.stop()


class VLLMDockerRunner(InferenceEngineRunner):
    def __init__(self,
                 model: str,
                 image: str = "vllm/vllm-openai:latest",
                 volumes=None):
        super().__init__(model)
        if volumes is None:
            volumes = []
        self.container = None
        self.image = image
        self.volumes = volumes

    def run(self, parameters: list[tuple]):
        parameters.append(("max-num-seqs", "256"))
        params = f"--model {self.model} --tensor-parallel-size {get_num_gpus()} --port 8080"
        for p in parameters:
            params += f" --{p[0]} {str(p[1])}"
        logger.info(f"Running VLLM with parameters: {params}")
        volumes = {}
        for v in self.volumes:
            volumes[v[0]] = {"bind": v[1], "mode": "rw"}
        self.container = run_docker(self.image, params, "Uvicorn running",
                                    "Error ",
                                    volumes=volumes)

    def stop(self):
        if self.container:
            self.container.stop()


def run_docker(image: str, args: str, success_sentinel: str,
               error_sentinel: str, volumes=None) -> Container:
    if volumes is None:
        volumes = {}
    client = docker.from_env()
    # retrieve the GPU devices from CUDA_VISIBLE_DEVICES
    devices = [f"{i}" for i in
               range(get_num_gpus())]
    container = client.containers.run(image, args,
                                      detach=True,
                                      device_requests=[
                                          docker.types.DeviceRequest(device_ids=devices, capabilities=[['gpu']])
                                      ],
                                      volumes=volumes,
                                      shm_size="1g",
                                      ports={"8080/tcp": 8080})
    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")
        if success_sentinel.encode("utf-8") in line:
            break
        if error_sentinel.encode("utf-8") in line:
            container.stop()
            raise Exception(f"Error starting container: {line}")
    return container


def get_num_gpus() -> int:
    return len(subprocess.run(["nvidia-smi", "-L"], capture_output=True).stdout.splitlines())
