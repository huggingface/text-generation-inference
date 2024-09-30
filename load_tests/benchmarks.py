import json
import os
import traceback

import GPUtil
import docker
from docker.models.containers import Container
from loguru import logger
import pandas as pd


class InferenceEngineRunner:
    def __init__(self, model: str):
        self.model = model

    def run(self, parameters: list[tuple]):
        NotImplementedError("This method should be implemented by the subclass")

    def stop(self):
        NotImplementedError("This method should be implemented by the subclass")


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
                                    "ERROR",
                                    volumes=volumes)

    def stop(self):
        if self.container:
            self.container.stop()


class BenchmarkRunner:
    def __init__(self,
                 image: str = "ghcr.io/huggingface/text-generation-inference-benchmark:latest",
                 volumes=None):
        if volumes is None:
            volumes = []
        self.container = None
        self.image = image
        self.volumes = volumes

    def run(self, parameters: list[tuple]):
        params = ""
        for p in parameters:
            params += f" --{p[0]} {str(p[1])}" if p[1] is not None else f" --{p[0]}"
        logger.info(f"Running text-generation-inference-benchmarks with parameters: {params}")
        volumes = {}
        for v in self.volumes:
            volumes[v[0]] = {"bind": v[1], "mode": "rw"}
        self.container = run_docker(self.image, params,
                                    "Benchmark finished",
                                    "Error",
                                    volumes=volumes)

    def stop(self):
        if self.container:
            self.container.stop()


def run_docker(image: str, args: str, success_sentinel: str,
               error_sentinel: str, volumes=None, gpus: int = 0) -> Container:
    if volumes is None:
        volumes = {}
    client = docker.from_env()
    # retrieve the GPU devices from CUDA_VISIBLE_DEVICES
    devices = [f"{i}" for i in
               range(get_num_gpus())][:gpus]
    container = client.containers.run(image, args,
                                      detach=True,
                                      device_requests=[
                                          docker.types.DeviceRequest(device_ids=devices,
                                                                     capabilities=[['gpu']]) if gpus > 0 else None
                                      ],
                                      volumes=volumes,
                                      shm_size="1g",
                                      ports={"8080/tcp": 8080},
                                      environment={"HF_TOKEN": os.environ.get("HF_TOKEN")}, )
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
        return ''
    return f'{len(gpus)}x{gpus[0].name if gpus else "No GPU available"}'


def get_gpu_name() -> str:
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        return ''
    return gpus[0].name


def get_num_gpus() -> int:
    return len(GPUtil.getGPUs())


def build_df(model: str, data_files: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()
    # Load the results
    for key, filename in data_files.items():
        with open(filename, 'r') as f:
            data = json.load(f)
            for result in data['results']:
                entry = result
                [config] = pd.json_normalize(result['config']).to_dict(orient='records')
                entry.update(config)
                entry['engine'] = data['config']['meta']['engine']
                entry['tp'] = data['config']['meta']['tp']
                entry['version'] = data['config']['meta']['version']
                entry['model'] = model
                del entry['config']
                df = pd.concat([df, pd.DataFrame(entry, index=[0])])
    return df


def main():
    results_dir = 'results'
    logger.info('Starting benchmark')
    models = [
        ('meta-llama/Llama-3.1-8B-Instruct', 1),
        # ('meta-llama/Llama-3.1-70B-Instruct', 4),
        # ('mistralai/Mixtral-8x7B-Instruct-v0.1', 2),
    ]
    sha = os.environ.get('GITHUB_SHA')
    # create results directory
    os.makedirs(results_dir, exist_ok=True)
    for model in models:
        tgi_runner = TGIDockerRunner(model[0])
        runner = BenchmarkRunner(
            volumes=['results', '/opt/text-generation-inference-benchmark/results']
        )
        try:
            tgi_runner.run([('max-concurrent-requests', 512)])
            logger.info(f'TGI started for model {model[0]}')
            parameters = [
                ('tokenizer-name', model[0]),
                ('max-vus', 800),
                ('url', 'http://localhost:8080'),
                ('duration', '120s'),
                ('warmup', '30s'),
                ('benchmark-kind', 'rate'),
                ('prompt-options', 'num_tokens=200,max_tokens=220,min_tokens=180,variance=10'),
                ('decode-options', 'num_tokens=200,max_tokens=220,min_tokens=180,variance=10'),
                ('extra-meta', f'engine=TGI,tp={model[1]},version={sha},gpu={get_gpu_name()}'),
                ('--no-console', None)
            ]
            runner.run(parameters)
        except Exception as e:
            logger.error(f'Error running benchmark for model {model[0]}: {e}')
            # print the stack trace
            print(traceback.format_exc())
        finally:
            tgi_runner.stop()
            runner.stop()
    # list json files in results directory
    data_files = {}
    df = pd.DataFrame()
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            data_files[filename.split('.')[-2]] = f'{results_dir}/{filename}'
    df = pd.concat([df, build_df(results_dir.split('/')[-1], data_files)])
    df['device'] = get_gpu_name()
    df['error_rate'] = df['failed_requests'] / (df['failed_requests'] + df['successful_requests']) * 100.0
    df.to_parquet('s3://text-generation-inference-ci/benchmarks/ci/')


if __name__ == "__main__":
    main()
