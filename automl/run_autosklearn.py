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


def get_commands(dataset_ids):
    path = create_directory(os.path.join("resources", "auto-sklearn"))
    return [
        (
            f"python automl/auto_sklearn/main.py --id {id}",
            os.path.join(
                create_directory(
                    os.path.join(path, openml.datasets.get_dataset(id).name)
                ),
                f"std_out.txt",
            ),
            os.path.join(
                create_directory(
                    os.path.join(path, openml.datasets.get_dataset(id).name)
                ),
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


dataset_ids = ["40983", "40499", "1485", "1478", "1590"]
commands = get_commands(dataset_ids)

with tqdm(total=len(dataset_ids)) as pbar:
    with Pool(len(dataset_ids)) as pool:
        # Assign the commands (tasks) to the pool, and run it
        for _ in pool.istarmap(run_cmd, commands):
            pbar.update()
