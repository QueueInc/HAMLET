from functools import partial

import argparse

import numpy as np
from numpy import dtype

import pandas as pd

import ray
from ray import tune
from ray.tune.suggest.flaml import BlendSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from experiment.datasets import get_dataset
from experiment.objective import objective
from experiment.space import get_space
from hamlet.Buffer import Buffer


import psutil

psutil_memory_in_bytes = psutil.virtual_memory().total

ray._private.utils.get_system_memory = lambda: psutil_memory_in_bytes


def objective(x, a, b):
    return a * (x**0.5) + b


def trainable(config, ciao=None):
    # config (dict): A dict of hyperparameters.

    final_score = 0
    for x in range(20):
        final_score = objective(x, config["a"], config["b"])

    return {"accuracy": final_score}  # This sends the score to Tune.


analysis = tune.run(
    run_or_experiment=tune.with_parameters(trainable, ciao="cacca"),
    num_samples=2,
    config={"a": tune.choice([2, 3]), "b": tune.choice([2, 3])},
    metric="accuracy",
    mode="max",
)

print(analysis.dataframe())
