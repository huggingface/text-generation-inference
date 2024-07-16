import json
import os
from enum import Enum
import pandas as pd
from loguru import logger

from matplotlib import pyplot as plt
import scienceplots

plt.style.use('science')


class TestType(Enum):
    CONSTANT_VUS = "constant_vus"
    CONSTANT_ARRIVAL_RATE = "constant_arrival_rate"


def parse_json_files(directory: str, test_type: TestType) -> pd.DataFrame:
    metrics_to_keep = {'inter_token_latency': {'y': 'Time (ms)'}, 'end_to_end_latency': {'y': 'Time (ms)'},
                       'time_to_first_token': {'y': 'Time (ms)'}, 'tokens_throughput': {'y': 'Tokens/s'},
                       'tokens_received': {'y': 'Count'}}
    df = pd.DataFrame()
    for file in os.listdir(directory):
        if file.endswith("summary.json"):
            filepath = os.path.join(directory, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if test_type == TestType.CONSTANT_VUS:
                    entry = {
                        "vus": data['k6_config']['vus'],
                        "duration": data['k6_config']['duration']
                    }
                elif test_type == TestType.CONSTANT_ARRIVAL_RATE:
                    entry = {
                        'pre_allocated_vus': data['k6_config']['pre_allocated_vus'],
                        'rate': data['k6_config']['rate'],
                        'duration': data['k6_config']['duration']
                    }
                entry['input_type'] = data['k6_config']['input_type']
                entry['test_duration'] = data['state']['testRunDurationMs'] / 1000.
                entry['requests_ok'] = data['root_group']['checks'][0]['passes']
                entry['requests_fail'] = data['root_group']['checks'][0]['fails']
                entry['dropped_iterations'] = data['metrics']['dropped_iterations']['values'][
                    'count'] if 'dropped_iterations' in data['metrics'] else 0
                # add up requests_fail and dropped_iterations to get total dropped requests
                entry['dropped_requests'] = entry['requests_fail'] + entry['dropped_iterations']
                entry['error_rate'] = entry['dropped_requests'] / (
                        entry['requests_ok'] + entry['dropped_requests']) * 100.0
                entry['name'] = data['k6_config']['name']
                for metric, values in sorted(data['metrics'].items()):
                    if metric in metrics_to_keep:
                        for value_key, value in values['values'].items():
                            if value_key == 'p(90)' or value_key == 'count':  # Only keep p(90) values if trend
                                entry[metric] = value
                if 'tokens_throughput' in entry and 'test_duration' in entry:
                    entry['tokens_throughput'] = entry['tokens_throughput'] / (entry['test_duration'])
                if 'inter_token_latency' in entry:
                    entry['inter_token_latency'] = entry['inter_token_latency'] / 1000.
                df = pd.concat([df, pd.DataFrame(entry, index=[0])])
    return df


def plot_metrics(model_name:str, df: pd.DataFrame, test_type: TestType, save_name: str):
    vus_param = ''
    if test_type == TestType.CONSTANT_VUS:
        vus_param = 'vus'
    else:
        vus_param = 'rate'
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    fig.tight_layout(pad=6.0)
    fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.15, top=0.92)

    names = sorted(df['name'].unique())
    metrics = {'inter_token_latency': {'y': 'Time (ms)'}, 'time_to_first_token': {'y': 'Time (ms)'},
               'end_to_end_latency': {'y': 'Time (ms)'}, 'tokens_throughput': {'y': 'Tokens/s'},
               'requests_ok': {'y': 'Count'}, 'error_rate': {'y': 'Count'}}
    titles = ['Inter Token Latency P90 (lower is better)', 'TTFT P90 (lower is better)',
              'End to End Latency P90 (lower is better)', 'Request Output Throughput P90 (higher is better)',
              'Successful requests (higher is better)', 'Error rate (lower is better)']
    labels = ['Time (ms)', 'Time (ms)', 'Time (ms)', 'Tokens/s', 'Count', '%']
    colors = ['#FF9D00', '#2F5BA1']
    # Plot each metric in its respective subplot
    for ax, metric, title, label in zip(axs.flatten(), metrics, titles, labels):
        for i, name in enumerate(names):
            df_sorted = df[df['name'] == name].sort_values(by=vus_param)
            ax.plot(df_sorted[vus_param], df_sorted[metric], marker='o', label=f"{name}", color=colors[i])
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=0)
            ax.set_ylabel(label)
            if test_type == TestType.CONSTANT_VUS:
                ax.set_xlabel('VUS')
            else:
                ax.set_xlabel('Requests/s')
            # Add grid lines for better readability
            ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)  # Ensure grid lines are below the bars
            ax.legend(title='Engine', loc='upper right')

    # show title on top of the figure
    if test_type == TestType.CONSTANT_VUS:
        plt.suptitle(f'Constant VUs Load Test\n{model_name}', fontsize=16)
    elif test_type == TestType.CONSTANT_ARRIVAL_RATE:
        plt.suptitle(f'Constant Arrival Rate Load Test\n{model_name}', fontsize=16)
    logger.info(f"Saving plot to {save_name}.png")
    plt.savefig(f"{save_name}.png")


def main():
    for test_type in [TestType.CONSTANT_VUS, TestType.CONSTANT_ARRIVAL_RATE]:
        directory = f"results/{test_type.value.lower()}"
        dfs = parse_json_files(directory, test_type)
        plot_metrics(dfs, test_type, test_type.value.lower())


if __name__ == "__main__":
    main()
