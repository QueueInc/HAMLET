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

    # TODO: all these things in the params
    metric = "accuracy"
    seed = 42
    np.random.seed(seed)
    # keep it in mind that this is incremental iterations after iterations (past points_to_evaluate and instance_constraints should be included in the count)
    batch_size = 25

    buffer = Buffer(args["path"])
    space = buffer.get_space()
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()

    print(
        pd.concat(
            [
                pd.DataFrame(points_to_evaluate),
                pd.DataFrame(
                    [reward["accuracy"] for reward in evaluated_rewards],
                    columns=["accuracy"],
                ),
            ],
            axis=1,
        )
    )

    analysis = tune.run(
        evaluation_function=partial(objective, X, y, metric, seed),
        config=space,
        metric=metric,
        mode="max",
        num_samples=batch_size,
        points_to_evaluate=points_to_evaluate,
        # evaluated_rewards=evaluated_rewards,
        verbose=False,
    )

    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    print(
        pd.concat(
            [
                pd.DataFrame(points_to_evaluate),
                pd.DataFrame(
                    [reward["accuracy"] for reward in evaluated_rewards],
                    columns=["accuracy"],
                ),
            ],
            axis=1,
        )
    )

    pd.concat(
        [
            pd.DataFrame(points_to_evaluate),
            pd.DataFrame(
                [reward["accuracy"] for reward in evaluated_rewards],
                columns=["accuracy"],
            ),
        ],
        axis=1,
    ).to_csv("trial.csv")


if __name__ == "__main__":
    # args = parse_args()
    args = {"path": "/workspace/knowledge.json"}
    main(args)
