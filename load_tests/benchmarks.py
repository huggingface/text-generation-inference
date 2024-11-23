import argparse
import datetime
import json
import os
import traceback
from typing import Dict, Tuple, List

import GPUtil
import docker
from docker.models.containers import Container
from loguru import logger
import pandas as pd


class InferenceEngineRunner:
    def __init__(self, model: str):
        self.model = model

    def run(self, parameters: list[tuple], gpus: int = 0):
        NotImplementedError("This method should be implemented by the subclass")

    def stop(self):
        NotImplementedError("This method should be implemented by the subclass")


class TGIDockerRunner(InferenceEngineRunner):
    def __init__(
        self,
        model: str,
        image: str = "ghcr.io/huggingface/text-generation-inference:latest",
        volumes=None,
    ):
        super().__init__(model)
        if volumes is None:
            volumes = []
        self.container = None
        self.image = image
        self.volumes = volumes

    def run(self, parameters: list[tuple], gpus: int = 0):
        params = f"--model-id {self.model} --port 8080"
        for p in parameters:
            params += f" --{p[0]} {str(p[1])}"
        logger.info(f"Running TGI with parameters: {params}")
        volumes = {}
        for v in self.volumes:
            volumes[v[0]] = {"bind": v[1], "mode": "rw"}
        self.container = run_docker(
            self.image,
            params,
            "Connected",
            "ERROR",
            volumes=volumes,
            gpus=gpus,
            ports={"8080/tcp": 8080},
        )

    def stop(self):
        if self.container:
            self.container.stop()


class BenchmarkRunner:
    def __init__(
        self,
        image: str = "ghcr.io/huggingface/text-generation-inference-benchmark:latest",
        volumes: List[Tuple[str, str]] = None,
    ):
        if volumes is None:
            volumes = []
        self.container = None
        self.image = image
        self.volumes = volumes

    def run(self, parameters: list[tuple], network_mode):
        params = "text-generation-inference-benchmark"
        for p in parameters:
            params += f" --{p[0]} {str(p[1])}" if p[1] is not None else f" --{p[0]}"
        logger.info(
            f"Running text-generation-inference-benchmarks with parameters: {params}"
        )
        volumes = {}
        for v in self.volumes:
            volumes[v[0]] = {"bind": v[1], "mode": "rw"}
        self.container = run_docker(
            self.image,
            params,
            "Benchmark finished",
            "Fatal:",
            volumes=volumes,
            extra_env={
                "RUST_LOG": "text_generation_inference_benchmark=info",
                "RUST_BACKTRACE": "full",
            },
            network_mode=network_mode,
        )

    def stop(self):
        if self.container:
            self.container.stop()


def run_docker(
    image: str,
    args: str,
    success_sentinel: str,
    error_sentinel: str,
    ports: Dict[str, int] = None,
    volumes=None,
    network_mode: str = "bridge",
    gpus: int = 0,
    extra_env: Dict[str, str] = None,
) -> Container:
    if ports is None:
        ports = {}
    if volumes is None:
        volumes = {}
    if extra_env is None:
        extra_env = {}
    client = docker.from_env(timeout=300)
    # retrieve the GPU devices from CUDA_VISIBLE_DEVICES
    devices = [f"{i}" for i in range(get_num_gpus())][:gpus]
    environment = {"HF_TOKEN": os.environ.get("HF_TOKEN")}
    environment.update(extra_env)
    container = client.containers.run(
        image,
        args,
        detach=True,
        device_requests=(
            [docker.types.DeviceRequest(device_ids=devices, capabilities=[["gpu"]])]
            if gpus > 0
            else None
        ),
        volumes=volumes,
        shm_size="1g",
        ports=ports,
        network_mode=network_mode,
        environment=environment,
    )
    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")
        if success_sentinel.encode("utf-8") in line:
            break
        if error_sentinel.encode("utf-8") in line:
            container.stop()
            raise Exception(f"Error starting container: {line}")
    return container


def get_gpu_names() -> str:
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        return ""
    return f'{len(gpus)}x{gpus[0].name if gpus else "No GPU available"}'


def get_gpu_name() -> str:
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        return ""
    return gpus[0].name


def get_num_gpus() -> int:
    return len(GPUtil.getGPUs())


def build_df(model: str, data_files: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()
    now = datetime.datetime.now(datetime.timezone.utc)
    created_at = now.isoformat()  # '2024-10-02T11:53:17.026215+00:00'
    # Load the results
    for key, filename in data_files.items():
        with open(filename, "r") as f:
            data = json.load(f)
            for result in data["results"]:
                entry = result
                [config] = pd.json_normalize(result["config"]).to_dict(orient="records")
                entry.update(config)
                entry["engine"] = data["config"]["meta"]["engine"]
                entry["tp"] = data["config"]["meta"]["tp"]
                entry["version"] = data["config"]["meta"]["version"]
                entry["model"] = model
                entry["created_at"] = created_at
                del entry["config"]
                df = pd.concat([df, pd.DataFrame(entry, index=[0])])
    return df


def main(sha, results_file):
    results_dir = "results"
    # get absolute path
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)
    logger.info("Starting benchmark")
    models = [
        ("meta-llama/Llama-3.1-8B-Instruct", 1),
        # ('meta-llama/Llama-3.1-70B-Instruct', 4),
        # ('mistralai/Mixtral-8x7B-Instruct-v0.1', 2),
    ]
    success = True
    for model in models:
        tgi_runner = TGIDockerRunner(model[0])
        # create results directory
        model_dir = os.path.join(
            results_dir, f'{model[0].replace("/", "_").replace(".", "_")}'
        )
        os.makedirs(model_dir, exist_ok=True)
        runner = BenchmarkRunner(
            volumes=[(model_dir, "/opt/text-generation-inference-benchmark/results")]
        )
        try:
            tgi_runner.run([("max-concurrent-requests", 512)], gpus=model[1])
            logger.info(f"TGI started for model {model[0]}")
            parameters = [
                ("tokenizer-name", model[0]),
                ("max-vus", 800),
                ("url", "http://localhost:8080"),
                ("duration", "120s"),
                ("warmup", "30s"),
                ("benchmark-kind", "rate"),
                (
                    "prompt-options",
                    "num_tokens=200,max_tokens=220,min_tokens=180,variance=10",
                ),
                (
                    "decode-options",
                    "num_tokens=200,max_tokens=220,min_tokens=180,variance=10",
                ),
                (
                    "extra-meta",
                    f'"engine=TGI,tp={model[1]},version={sha},gpu={get_gpu_name()}"',
                ),
                ("no-console", None),
            ]
            rates = [("rates", f"{r / 10.}") for r in list(range(8, 248, 8))]
            parameters.extend(rates)
            runner.run(parameters, f"container:{tgi_runner.container.id}")
        except Exception as e:
            logger.error(f"Error running benchmark for model {model[0]}: {e}")
            # print the stack trace
            print(traceback.format_exc())
            success = False
        finally:
            tgi_runner.stop()
            runner.stop()
    if not success:
        logger.error("Some benchmarks failed")
        exit(1)

    df = pd.DataFrame()
    # list recursively directories
    directories = [
        f"{results_dir}/{d}"
        for d in os.listdir(results_dir)
        if os.path.isdir(f"{results_dir}/{d}")
    ]
    logger.info(f"Found result directories: {directories}")
    for directory in directories:
        data_files = {}
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                data_files[filename.split(".")[-2]] = f"{directory}/{filename}"
        logger.info(f"Processing directory {directory}")
        df = pd.concat([df, build_df(directory.split("/")[-1], data_files)])
    df["device"] = get_gpu_name()
    df["error_rate"] = (
        df["failed_requests"]
        / (df["failed_requests"] + df["successful_requests"])
        * 100.0
    )
    df.to_parquet(results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sha", help="SHA of the commit to add to the results", required=True
    )
    parser.add_argument(
        "--results-file",
        help="The file where to store the results, can be a local file or a s3 path",
    )
    args = parser.parse_args()
    if args.results_file is None:
        results_file = f"{args.sha}.parquet"
    else:
        results_file = args.results_file

    main(args.sha, results_file)
