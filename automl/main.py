from functools import partial
from hamlet.miner import Miner

from utils.argparse import parse_args
from utils.json_to_csv import json_to_csv

import json

import numpy as np
from numpy import dtype

import pandas as pd

from flaml import tune

from utils.datasets import get_dataset
from hamlet.objective import objective
from hamlet.buffer import Buffer


def main(args):
    np.random.seed(args.seed)

    X, y, _ = get_dataset(args.dataset)
    buffer = Buffer(metric=args.metric, input_path=args.input_path)
    space = buffer.get_space()
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()

    # print(
    #     pd.concat(
    #         [
    #             pd.DataFrame(points_to_evaluate),
    #             pd.DataFrame(
    #                 [reward[args.metric] for reward in evaluated_rewards],
    #                 columns=[args.metric],
    #             ),
    #         ],
    #         axis=1,
    #     )
    # )

    tune.run(
        evaluation_function=partial(objective, X, y, args.metric, args.seed),
        config=space,
        metric=args.metric,
        mode=args.mode,
        num_samples=args.batch_size + len(points_to_evaluate),
        points_to_evaluate=points_to_evaluate,
        # evaluated_rewards=evaluated_rewards,
        verbose=False,
    )

    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    points_to_evaluate = points_to_evaluate[-buffer.get_num_points_to_consider() :]
    evaluated_rewards = evaluated_rewards[-buffer.get_num_points_to_consider() :]

    miner = Miner(
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        metric=args.metric,
        mode=args.mode,
    )
    rules = miner.get_rules()

    automl_output = {
        "points_to_evaluate": points_to_evaluate,
        "evaluated_rewards": [str(reward[args.metric]) for reward in evaluated_rewards],
        "rules": rules,
    }

    with open(args.output_path, "w") as outfile:
        json.dump(automl_output, outfile)

    json_to_csv(automl_output=automl_output.copy(), args=args)

    # results_df = pd.concat(
    #     [
    #         pd.DataFrame(points_to_evaluate[-buffer.get_num_points_to_consider():]),
    #         pd.DataFrame(
    #             [reward[args.metric] for reward in evaluated_rewards[-buffer.get_num_points_to_consider():]],
    #             columns=[args.metric],
    #         ),
    #     ],
    #     axis=1,
    # )
    # print(results_df)
    # results_df.to_csv("automl_output.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
