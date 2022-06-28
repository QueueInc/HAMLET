import subprocess
import openml
import os
import argparse

from multiprocessing import Pool
from tqdm import tqdm


def get_commands(data, args):
    commands = []
    for dataset in data:
        dataset_path = os.path.join(os.getcwd(), args.workspace, str(dataset))
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
        commands.append((cmd, stdout_path, stderr_path))
    return commands


def run_cmd(cmd, stdout_path, stderr_path):
    open(stdout_path, "w")
    open(stderr_path, "w")
    with open(stdout_path, "a") as log_out:
        with open(stderr_path, "a") as log_err:
            subprocess.call(cmd, stdout=log_out, stderr=log_err, bufsize=0, shell=True)


def create_directory(result_path, directory):
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path


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
    parser.add_argument(
        "-range",
        "--range",
        nargs="?",
        type=int,
        required=False,
        help="which index of the suite to start",
    )
    parser.add_argument(
        "-num_tasks",
        "--num_tasks",
        nargs="?",
        type=int,
        required=False,
        help="which index of the suite to start",
    )
    args = parser.parse_args()
    return args


args = parse_args()
data = openml.study.get_suite("OpenML-CC18").data
data = data[args.range : args.range + int(len(data) / args.num_tasks)]
commands = get_commands(data, args)


with tqdm(total=len(data)) as pbar:
    for cmd, stdout_path, stderr_path in get_commands(data, args):
        run_cmd(cmd, stdout_path, stderr_path)
        pbar.update()
