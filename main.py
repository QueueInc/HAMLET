from functools import partial

# Numpy provides useful data structure
import numpy as np
from numpy import dtype

# Pandas provides a tabular view of the data
import pandas as pd

# FLAML provides Bayesian Optimization techniques
from flaml import tune

from experiment.datasets import get_dataset
from experiment.objective import objective
from experiment.space import get_space
from hamlet.Buffer import Buffer


def main():
    # We load the wine dataset
    X, y, _ = get_dataset("wine")

    space = get_space()
    metric = "accuracy"
    seed = 42
    np.random.seed(seed)
    batch_size = 10

    buffer = Buffer()
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()

    # We run SMBO for each of the optimizations
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
    # execute only if run as the entry point into the program
    main()
