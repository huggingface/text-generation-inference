import os
import time
import traceback

from benchmarks.engine import TGIDockerRunner
from benchmarks.k6 import K6Config, K6Benchmark, K6ConstantArrivalRateExecutor, K6ConstantVUsExecutor, ExecutorInputType
from loguru import logger
import pandas as pd
import GPUtil

from parse_load_test import TestType, parse_json_files, plot_metrics


def run_full_test(engine_name: str):
    vus_concurrences = list(range(0, 1024, 40))
    vus_concurrences[0] = 1
    vus_concurrences.append(1024)
    arrival_rates = list(range(0, 200, 10))
    arrival_rates[0] = 1
    arrival_rates.append(200)
    for input_type in [ExecutorInputType.SHAREGPT_CONVERSATIONS, ExecutorInputType.CONSTANT_TOKENS]:
        for c in arrival_rates:
            logger.info(f'Running k6 with constant arrival rate for {c} req/s with input type {input_type.value}')
            k6_executor = K6ConstantArrivalRateExecutor(2000, c, '60s', input_type)
            k6_config = K6Config(f'{engine_name}', k6_executor, input_num_tokens=200)
            benchmark = K6Benchmark(k6_config, f'results/{input_type.value}/')
            benchmark.run()
        for c in vus_concurrences:
            logger.info(f'Running k6 with constant VUs with concurrency {c} with input type {input_type.value}')
            k6_executor = K6ConstantVUsExecutor(c, '60s', input_type)
            k6_config = K6Config(f'{engine_name}', k6_executor, input_num_tokens=200)
            benchmark = K6Benchmark(k6_config, f'results/{input_type.value}/')
            benchmark.run()


def merge_previous_results(csv_path: str, df: pd.DataFrame, version_id: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        previous_df = pd.read_csv(csv_path)
        previous_df['name'] = previous_df['name'].str.replace('tgi', f'tgi_{version_id}')
        df = pd.concat([previous_df, df])
    return df


def main():
    model = 'Qwen/Qwen2-7B'
    runner = TGIDockerRunner(model)
    max_concurrent_requests = 8000
    # run TGI
    try:
        logger.info('Running TGI')
        runner.run([('max-concurrent-requests', max_concurrent_requests)])
        logger.info('TGI is running')
        run_full_test('tgi')
    except Exception as e:
        logger.error(f'Error: {e}')
        # print the stack trace
        print(traceback.format_exc())
    finally:
        runner.stop()
        time.sleep(5)

    for input_type in [ExecutorInputType.SHAREGPT_CONVERSATIONS, ExecutorInputType.CONSTANT_TOKENS]:
        for test_type in [TestType.CONSTANT_VUS, TestType.CONSTANT_ARRIVAL_RATE]:
            directory = os.path.join('results', input_type.value.lower(), test_type.value.lower())
            # check if directory exists
            if not os.path.exists(directory):
                logger.error(f'Directory {directory} does not exist')
                continue
            dfs = parse_json_files(directory, test_type)
            # create output directory if it does not exist
            os.makedirs('output', exist_ok=True)
            # save the data to a csv file
            path = os.path.join(os.getcwd(), 'output', f'{input_type.value.lower()}_{test_type.value.lower()}.csv')
            dfs.to_csv(path)
            # check if we have previous results CSV file by listing /tmp/artifacts/<input_type> directory,
            # merge them if they exist
            prev_root = '/tmp/artifacts'
            try:
                if os.path.exists(prev_root):
                    directories = [item for item in os.listdir(prev_root) if
                                   os.path.isdir(os.path.join(prev_root, item))]
                    for d in directories:
                        for f in os.listdir(f'{prev_root}/{d}'):
                            if f.endswith(f'{input_type.value.lower()}_{test_type.value.lower()}.csv'):
                                csv_path = os.path.join('/tmp/artifacts', d, f)
                                # only keep short commit hash
                                d = d[:7]
                                dfs = merge_previous_results(csv_path, dfs, d)
            except Exception as e:
                logger.error(f'Error while merging previous results, skipping: {e}')
            plot_metrics(f'{model} {get_gpu_names()}', dfs, test_type,
                         f'output/{input_type.value.lower()}_{test_type.value.lower()}')


def get_gpu_names() -> str:
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        return ''
    return f'{len(gpus)}x{gpus[0].name if gpus else "No GPU available"}'


if __name__ == '__main__':
    main()
