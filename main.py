from asyncore import write
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
from ray.tune.suggest.skopt import SkOptSearch

from experiment.datasets import get_dataset
from experiment.objective import objective
from experiment.space import get_space
from hamlet.Buffer import Buffer

from skopt import Optimizer

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
    algo = HyperOptSearch(
        metric=metric, mode="max"  # , points_to_evaluate=points_to_evaluate[:20]
    )
    # algo = BlendSearch()
    algo = ConcurrencyLimiter(algo, 1)
    analysis = tune.run(
        run_or_experiment=tune.with_parameters(
            objective, X=X, y=y, metric="accuracy", seed=seed
        ),
        metric="accuracy",
        mode="max",
        config=space,
        num_samples=20,
        # stop={"training_iteration": batch_size},
        search_alg=algo,
        # evaluated_rewards=evaluated_rewards,
        verbose=0,
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

    pd.concat(
        [
            pd.DataFrame(points_to_evaluate),
            pd.DataFrame(
                [reward["accuracy"] for reward in evaluated_rewards],
                columns=["accuracy"],
            ),
        ],
        axis=1,
    ).to_csv("automl_output.csv")


def prova():
    def objective(x, a, b):
        return a * (x**0.5) + b

    def trainable(config, checkpoint_dir=None, pippo=None):
        print(pippo)
        # config (dict): A dict of hyperparameters.
        final_score = 0
        for x in range(20):
            final_score = objective(x, config["a"], config["b"])
            tune.report(
                **{"merda": final_score, "ciao": 3}
            )  # This sends the score to Tune.

    # algo = BlendSearch()

    # optimizer = Optimizer(
    #     base_estimator="rf",
    # )

    hyperopt_search = HyperOptSearch(metric="merda", mode="min")

    # skopt_search = SkOptSearch(metric="merda", mode="min")

    analysis = tune.run(
        tune.with_parameters(trainable, pippo="ciao"),
        # trainable,
        config={
            "a": tune.randint(1, 20),
            "b": tune.randint(1, 20),
            "c": tune.choice([{"d": "a", "a": tune.randint(1, 10)}]),
        },
        num_samples=4,
        metric="merda",
        mode="min",
        search_alg=hyperopt_search,
    )

    print("best config: ", analysis.get_best_config(metric="merda", mode="max"))


if __name__ == "__main__":
    # args = parse_args()
    args = {"path": "/workspace/knowledge-generated.json"}
    main(args)
    # prova()
