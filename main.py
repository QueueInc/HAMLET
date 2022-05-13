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

    buffer = Buffer(args["path"])
    space = buffer.get_space()
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()

    # keep it in mind that this is incremental iterations after iterations (past points_to_evaluate and instance_constraints should be included in the count)
    batch_size = 25 + len(points_to_evaluate)

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

    # algo = BayesOptSearch(
    #    metric=metric, mode="max", points_to_evaluate=points_to_evaluate
    # )
    # algo = HyperOptSearch(
    #    metric=metric, mode="max", points_to_evaluate=points_to_evaluate
    # )
    algo = BlendSearch(points_to_evaluate=[points_to_evaluate[-1]])
    algo = ConcurrencyLimiter(algo, 1)
    analysis = tune.run(
        run_or_experiment=tune.with_parameters(
            objective, X=X, y=y, metric=metric, seed=seed
        ),
        metric=metric,
        mode="max",
        config=space,
        num_samples=20,
        # stop={"training_iteration": batch_size},
        search_alg=algo,
        # evaluated_rewards=evaluated_rewards,
        # verbose=False,
    )

    # points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    # print(
    #     pd.concat(
    #         [
    #             pd.DataFrame(points_to_evaluate),
    #             pd.DataFrame(
    #                 [reward["accuracy"] for reward in evaluated_rewards],
    #                 columns=["accuracy"],
    #             ),
    #         ],
    #         axis=1,
    #     )
    # )

    print(analysis.dataframe())

    # pd.concat(
    #     [
    #         pd.DataFrame(points_to_evaluate),
    #         pd.DataFrame(
    #             [reward["accuracy"] for reward in evaluated_rewards],
    #             columns=["accuracy"],
    #         ),
    #     ],
    #     axis=1,
    # ).to_csv("automl_output.csv")


if __name__ == "__main__":
    # args = parse_args()
    args = {"path": "/workspace/knowledge-generated.json"}
    main(args)
