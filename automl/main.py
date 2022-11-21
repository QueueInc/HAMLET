import json
import time
import warnings
import sys

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from functools import partial
from flaml import tune

from utils.argparse import parse_args
from utils.json_to_csv import json_to_csv
from utils.datasets import get_dataset_by_id, get_dataset_by_name, load_from_csv
from hamlet.objective import objective
from hamlet.loader import Loader
from hamlet.buffer import Buffer
from hamlet.miner import Miner


def main(args):
    np.random.seed(args.seed)

    # X, y, _ = get_dataset_by_name(args.dataset)
    start_time = time.time()
    X, y, categorical_indicator = load_from_csv(args.dataset)
    loader = Loader(args.input_path)
    buffer = Buffer(metric=args.metric, loader=loader)
    space = loader.get_space()

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

    previous_evaluated_points = (
        loader.get_points_to_evaluate() + loader.get_instance_constraints()
    )
    graph_generation_time = loader.get_graph_generation_time()
    space_generation_time = loader.get_space_generation_time()
    del loader

    analysis = tune.run(
        evaluation_function=partial(
            objective, X, y, categorical_indicator, args.metric, args.seed
        ),
        config=space,
        metric=args.metric,
        mode=args.mode,
        num_samples=(args.batch_size + len(previous_evaluated_points))
        if args.batch_size > 0
        else -1,
        time_budget_s=args.time_budget if args.time_budget > 0 else None,
        points_to_evaluate=previous_evaluated_points,
        # evaluated_rewards=evaluated_rewards,
        verbose=0,
        max_failure=sys.maxsize * 2 + 1,  # args.batch_size + len(points_to_evaluate),
    )

    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()

    miner = Miner(
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        metric=args.metric,
        mode=args.mode,
    )
    end_time = time.time()
    rules = miner.get_rules()

    automl_output = {
        "start_time": start_time,
        "graph_generation_time": graph_generation_time,
        "space_generation_time": space_generation_time,
        "optimization_time": end_time - start_time,
        "mining_time": time.time() - end_time,
        "best_config": analysis.best_trial.last_result,
        "rules": rules,
        "points_to_evaluate": points_to_evaluate,
        # "evaluated_rewards": [str(reward[args.metric]) for reward in evaluated_rewards],
        "evaluated_rewards": [
            json.loads(str(reward).replace("'", '"').replace("-inf", '"-inf"'))
            for reward in evaluated_rewards
        ],
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
