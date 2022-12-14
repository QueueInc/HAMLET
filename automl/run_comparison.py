import subprocess
import openml
import math
import os
import argparse

import autosklearn.classification

import numpy as np
import pandas as pd

import utils.istarmap  # import to apply patch

from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def get_commands(dataset_ids, args):
    path = create_directory(args.output_folder)
    return [
        (
            f"python automl/{args.tool}/main.py --id {id} --budget {args.budget} --path {args.output_folder}",
            os.path.join(
                create_directory(os.path.join(path, id)),
                f"std_out.txt",
            ),
            os.path.join(
                create_directory(os.path.join(path, id)),
                f"std_err.txt",
            ),
        )
        for id in dataset_ids
    ]


def run_cmd(cmd, stdout_path, stderr_path):
    """Run a command in the shell, and save std out and std err.
    Args:
        cmd (string): the command to run
        stdout_path (str, bytes or os.PathLike object): where to save the std out.
        stderr_path (str, bytes or os.PathLike object): where to save the std err.
    """
    open(stdout_path, "w")
    open(stderr_path, "w")
    with open(stdout_path, "a") as log_out:
        with open(stderr_path, "a") as log_err:
            subprocess.call(cmd, stdout=log_out, stderr=log_err, bufsize=0, shell=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning Workflow creation and configuration"
    )
    parser.add_argument(
        "-tool",
        "--tool",
        nargs="?",
        type=str,
        required=True,
        help="Tool to test",
    )
    parser.add_argument(
        "-budget",
        "--budget",
        nargs="?",
        type=int,
        required=True,
        help="Time busget",
    )
    parser.add_argument(
        "-output_folder",
        "--output_folder",
        nargs="?",
        type=str,
        required=True,
        help="Time busget",
    )
    args = parser.parse_args()
    return args


args = parse_args()
dataset_ids = ["40983", "40499", "1485", "1478", "1590"]
commands = get_commands(dataset_ids, args)

with tqdm(total=len(dataset_ids)) as pbar:
    for cmd, stdout_path, stderr_path in commands:
        print(cmd)
        run_cmd(cmd, stdout_path, stderr_path)
        pbar.update()
