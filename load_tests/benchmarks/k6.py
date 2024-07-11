import json
import os
import subprocess
import tempfile
from enum import Enum
from typing import Any, Dict, List

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
from loguru import logger
from transformers import LlamaTokenizerFast

from benchmarks.utils import kill

env = Environment(
    loader=PackageLoader("benchmarks"),
    autoescape=select_autoescape()
)


class ExecutorInputType(Enum):
    CONSTANT_TOKENS = "constant_tokens"
    SHAREGPT_CONVERSATIONS = "sharegpt_conversations"


class K6Executor:
    def __init__(self, name, template_name, executor_input_type=ExecutorInputType.SHAREGPT_CONVERSATIONS):
        self.template_name = template_name
        self.variables = {}
        self.rendered_file = None
        self.name = name
        self.executor_input_type = executor_input_type
        if executor_input_type == ExecutorInputType.CONSTANT_TOKENS:
            self.input_filename = "inputs_constant_tokens.json"
        elif executor_input_type == ExecutorInputType.SHAREGPT_CONVERSATIONS:
            self.input_filename = "inputs_variable_tokens.json"

    def render(self):
        template = env.get_template(self.template_name)
        _, path = tempfile.mkstemp("k6", "benchmark")
        cwd = os.getcwd()
        with open(path, "w") as f:
            f.write(template.render(cwd=cwd, input_filename=self.input_filename, **self.variables))
        self.rendered_file = path

    def __str__(self):
        # returns an underscore separated string of the variables for filename generation
        params = "_".join([f"{k}_{v}" for k, v in sorted(self.variables.items()) if type(v) == str or type(v) == int])
        return f"{self.executor_input_type.value}_{params}"


class K6ConstantArrivalRateExecutor(K6Executor):
    def __init__(self, pre_allocated_vus: int, rate_per_second: int, duration: str,
                 executor_input_type: ExecutorInputType):
        super().__init__("constant_arrival_rate", "k6_constant_arrival_rate.js.j2", executor_input_type)
        self.variables = {
            "pre_allocated_vus": pre_allocated_vus,  # it's also the max vus
            "rate": rate_per_second,
            "duration": duration
        }


class K6RampingArrivalRateExecutor(K6Executor):
    def __init__(self, pre_allocated_vus: int, start_rate: int, time_unit: str, stages: List[Dict[str, Any]],
                 executor_input_type: ExecutorInputType):
        super().__init__("ramping_arrival_rate", "k6_ramping_arrival_rate.js.j2", executor_input_type)
        self.variables = {
            "pre_allocated_vus": pre_allocated_vus,
            "start_rate": start_rate,
            "time_unit": time_unit,
            "stages": stages
        }


class K6ConstantVUsExecutor(K6Executor):
    def __init__(self, vus: int, duration: str, executor_input_type: ExecutorInputType):
        super().__init__("constant_vus", "k6_constant_vus.js.j2", executor_input_type)
        self.variables = {
            "vus": vus,
            "duration": duration
        }


class K6Config:
    def __init__(self, name: str, executor: K6Executor,
                 tokenizer=LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer"),
                 conversations_input_file=None,
                 input_num_tokens=200,
                 max_new_tokens=200,
                 extra_info=None
                 ):
        self.executor = executor
        # max_new_token will be set in k6 template
        self.executor.variables["max_new_tokens"] = max_new_tokens
        self.name = name
        self.tokenizer = tokenizer
        self.extra_info = extra_info
        if conversations_input_file is None:
            self.conversation_input_file = "benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json"
        self.input_num_tokens = input_num_tokens

    def __str__(self):
        return f"K6Config(name={self.name} executor={self.executor})"


class K6Benchmark:
    def __init__(self, k6_config: K6Config, output_dir: str):
        self.process = None
        self.k6_config = k6_config
        self.output_dir = output_dir
        self.input_tokens_len = k6_config.input_num_tokens
        self._prepare_inputs()

    def _prepare_inputs(self):
        get_tokens_count = lambda txt: len(self.k6_config.tokenizer.encode(txt))
        MAX_SAMPLES = 5000

        # create a first input file with a constant number of tokens
        # check if the file already exists
        if not os.path.exists("inputs_constant_tokens.json"):
            logger.info(f'Preparing input file with {self.input_tokens_len} input tokens')
            outputs = []
            with open(self.k6_config.conversation_input_file, "r") as f:
                data = json.load(f)
                for doc in data:
                    for conversation in doc["conversations"]:
                        if not conversation["from"] == "human":
                            continue
                        if get_tokens_count(conversation["value"]) < self.input_tokens_len:
                            continue
                        # encode the message
                        encoding = self.k6_config.tokenizer(conversation["value"], truncation=True,
                                                            max_length=self.input_tokens_len)
                        # find last encoded characters
                        span = encoding.token_to_chars(len(encoding["input_ids"]) - 1)
                        outputs.append(
                            {"message": conversation["value"][0:span.end], "num_tokens": len(encoding["input_ids"])})
                    if len(outputs) >= MAX_SAMPLES:  # limit the number of inputs
                        break
            with open("inputs_constant_tokens.json", "w") as f:
                f.write(json.dumps(outputs))

        # create a second input file with a sampling of inputs
        # check if the file already exists
        if not os.path.exists("inputs_variable_tokens.json"):
            logger.info(
                f'Preparing input file by randomly sampling shareGPT conversations at "{self.k6_config.conversation_input_file}"')
            outputs = []
            with open(self.k6_config.conversation_input_file, "r") as f:
                data = json.load(f)
                num_docs = len(data)
                # generate random indexes to sample the data
                indexes = np.random.choice(num_docs, 200, replace=False)
                for i in indexes:
                    doc = data[i]
                    for conversation in doc["conversations"]:
                        if not conversation["from"] == "human":
                            continue
                        # encode the message without truncation
                        encoding = self.k6_config.tokenizer(conversation["value"])
                        outputs.append(
                            {"message": conversation["value"], "num_tokens": len(encoding["input_ids"])})
                    if len(outputs) >= MAX_SAMPLES:  # limit the number of inputs
                        break
            with open("inputs_variable_tokens.json", "w") as f:
                f.write(json.dumps(outputs))

    def run(self):
        self.k6_config.executor.render()
        args = f"/tmp/k6-sse run --out json=results.json {self.k6_config.executor.rendered_file}"
        logger.info(f"Running k6 with parameters: {args}")
        logger.info(f"K6Config is: {self.k6_config}")
        # start a k6 subprocess
        self.process = subprocess.Popen(args,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        while buffer := os.read(self.process.stdout.fileno(),
                                2048):  # read the output of the process, don't buffer on new lines
            print(buffer.decode(), end='')
        self.process.wait()
        logger.info(f"K6 process finished with return code {self.process.returncode}")
        logger.info(f"Writing results to {self.get_results_path()}")
        self.add_config_to_summary()
        self.add_config_to_results()

    def stop(self):
        if self.process:
            kill(self.process.pid)

    def add_config_to_summary(self):
        with open("summary.json", "r") as f:
            summary = json.load(f)
            summary["k6_config"] = {
                "name": self.k6_config.name,
                "input_type": self.k6_config.executor.executor_input_type.value,
                "extra_info": self.k6_config.extra_info,
                **self.k6_config.executor.variables
            }
            # create directory if it doesn't exist
            os.makedirs(self._get_output_dir(), exist_ok=True)
            with open(self.get_summary_path(), "w") as f2:
                json.dump(summary, f2)

    def add_config_to_results(self):
        with open("results.json", "r") as f:
            results = f.readlines()
            # append the k6 config to the results in jsonlines format
            results += "\n"
            results += json.dumps({
                "name": self.k6_config.name,
                "input_type": self.k6_config.executor.executor_input_type.value,
                "extra_info": self.k6_config.extra_info,
                **self.k6_config.executor.variables
            })
            # create directory if it doesn't exist
            os.makedirs(self._get_output_dir(), exist_ok=True)
            with open(self.get_results_path(), "w") as f2:
                f2.writelines(results)

    def _get_output_dir(self):
        # check if output_dir is relative or absolute
        if self.output_dir.startswith("/"):
            return f"{self.output_dir}/{self.k6_config.executor.name}"
        else:
            return f"{os.getcwd()}/{self.output_dir}/{self.k6_config.executor.name}"

    def _get_output_path(self):
        return f"{self._get_output_dir()}/{self.k6_config.name}_{self.k6_config.executor}"

    def get_results_path(self):
        return f"{self._get_output_path()}.json"

    def get_summary_path(self):
        return f"{self._get_output_path()}.summary.json"
