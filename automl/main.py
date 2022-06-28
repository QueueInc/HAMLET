import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from functools import partial
from flaml import tune

from utils.argparse import parse_args
from utils.json_to_csv import json_to_csv
from utils.datasets import get_dataset_by_id, get_dataset_by_name
from hamlet.objective import objective
from hamlet.buffer import Buffer
from hamlet.miner import Miner


def main(args):
    np.random.seed(args.seed)

    # X, y, _ = get_dataset_by_name(args.dataset)
    start_time = time.time()
    X, y, categorical_indicator = get_dataset_by_id(args.dataset)
    buffer = Buffer(metric=args.metric, input_path=args.input_path)
    space = buffer.loader.get_space()
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

    analysis = tune.run(
        evaluation_function=partial(
            objective, X, y, categorical_indicator, args.metric, args.seed
        ),
        config=space,
        metric=args.metric,
        mode=args.mode,
        num_samples=args.batch_size + len(points_to_evaluate),
        time_budget_s=14400,
        points_to_evaluate=points_to_evaluate,
        # evaluated_rewards=evaluated_rewards,
        verbose=0,
        max_failure=args.batch_size + len(points_to_evaluate),
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
    end_time = time.time()
    rules = miner.get_rules()

    automl_output = {
        "graph_generation_time": buffer.loader.get_graph_generation_time(),
        "space_generation_time": buffer.loader.get_space_generation_time(),
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
