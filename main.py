from functools import partial

import argparse

import numpy as np
from numpy import dtype

import pandas as pd

from flaml import tune

from experiment.datasets import get_dataset
from experiment.objective import objective
from experiment.space import get_space
from hamlet.Buffer import Buffer


def parse_args():

    parser = argparse.ArgumentParser(description="HAMLET")

    parser.add_argument(
        "-p", "--path", nargs="?", type=str, required=True, help="path to the knowledge"
    )

    args = parser.parse_args()
    return args


def main(args):
    X, y, _ = get_dataset("wine")

    metric = "accuracy"
    seed = 42
    np.random.seed(seed)
    batch_size = 10

    buffer = Buffer(args["path"])
    space = buffer.get_space()
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()

    # TODO:
    # gestire evaluated_rewards stateless
    # ottimizzare il check su evaluated_rewards e instance:constraints mettendo in una hasmap
    # gestire randint

    analysis = tune.run(
        evaluation_function=partial(objective, X, y, metric, seed),
        config=space,
        metric=metric,
        mode="max",
        num_samples=batch_size,
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        verbose=False,
    )

    print(analysis.best_result)


if __name__ == "__main__":
    # args = parse_args()
    args = {"path": "/workspace/knowledge.json"}
    main(args)
