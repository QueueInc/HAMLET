import subprocess
import openml
import os
import argparse

import pandas as pd

from tqdm import tqdm


def generate_processes(data):
    for dataset in data:
        dataset_path = os.path.join(workspace_path, str(dataset))
        log_path = create_directory(dataset_path, "logs")
        cmd = f"""java -jar hamlet-{args.version}-all.jar \
                    {dataset_path} \
                    {dataset} \
                    {args.metric} \
                    {args.mode} \
                    {args.batch_size} \
                    42 \
                    false \
                    $(pwd)/resources/complete_kb_5_steps.txt"""
        stdout_path = os.path.join(log_path, "stdout_1.txt")
        stderr_path = os.path.join(log_path, "stderr_1.txt")
        yield run_cmd(cmd=cmd, stdout_path=stdout_path, stderr_path=stderr_path)


def run_cmd(cmd, stdout_path, stderr_path):
    open(stdout_path, "w")
    open(stderr_path, "w")
    with open(stdout_path, "a") as log_out:
        with open(stderr_path, "a") as log_err:
            process = subprocess.Popen(
                cmd, shell=True, stdout=log_out, stderr=log_err, bufsize=0
            )
    return process


def create_directory(result_path, directory):
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path


def get_filtered_datasets(suite):
    df = pd.read_csv(os.path.join("temp", "simple-meta-features.csv"))
    df = df.loc[df["did"].isin(suite)]
    df = df.loc[
        df["NumberOfMissingValues"] / (df["NumberOfInstances"] * df["NumberOfFeatures"])
        < 0.1
    ]
    df = df.loc[
        df["NumberOfInstancesWithMissingValues"] / df["NumberOfInstances"] < 0.1
    ]
    df = df.loc[df["NumberOfInstances"] * df["NumberOfFeatures"] < 5000000]
    df = df["did"]
    return df.values.flatten().tolist()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning Workflow creation and configuration"
    )
    parser.add_argument(
        "-workspace",
        "--workspace",
        nargs="?",
        type=str,
        required=False,
        help="where to save the data",
    )
    parser.add_argument(
        "-metric",
        "--metric",
        nargs="?",
        type=str,
        required=False,
        help="metric to optimize",
    )
    parser.add_argument(
        "-mode",
        "--mode",
        nargs="?",
        type=str,
        required=False,
        help="how to optimize the metric",
    )
    parser.add_argument(
        "-batch_size",
        "--batch_size",
        nargs="?",
        type=str,
        required=False,
        help="automl confs to visit",
    )
    parser.add_argument(
        "-version",
        "--version",
        nargs="?",
        type=str,
        required=False,
        help="hamlet version to run",
    )
    args = parser.parse_args()
    return args


args = parse_args()
workspace_path = os.path.join(os.getcwd(), args.workspace)
processes = {}
benchmark_suite = openml.study.get_suite("OpenML-CC18")  # obtain the benchmark suite
data = get_filtered_datasets(benchmark_suite.data)
generator = generate_processes(data)

count = 0
max_processes = 10
buffer = []
is_buffer_free = True


with tqdm(total=len(data)) as pbar:
    while is_buffer_free:
        if count < max_processes:
            try:
                buffer.append(next(generator))
                count += 1
            except:
                is_buffer_free = len(buffer) != 0
                count = max_processes
        else:
            buffer.pop(0).wait()
            pbar.update(1)
            count -= 1
