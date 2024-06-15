import time
import numpy as np

from automl.hamlet.buffer import Buffer
from automl.hamlet.objective import Prototype
from automl.hamlet.loader import Loader
from automl.utils.datasets import load_dataset_from_openml
from automl.hamlet.engine import optimize, mine_results, dump_results

def run(args):

    np.random.seed(args.seed)
    metrics = [args.fair_metric, args.metric]

    X, y, categorical_indicator, sensitive_indicator = load_dataset_from_openml(
        args.dataset
    )
    
    loader = Loader(args.input_path)
    initial_design_configs = 5 if len(loader.get_points_to_evaluate()) == 0 else 0

    buffer = Buffer(
        metrics=metrics, loader=loader, initial_design_configs=initial_design_configs
    )

    buffer.attach_handler()
    start_time = time.time()

    Buffer().printflush("AutoML: starting optimization.")

    prototype = Prototype(
        X,
        y,
        categorical_indicator,
        sensitive_indicator,
        args.fair_metric,
        args.metric,
        args.mode,
    )

    _, _, best_config = optimize(args, prototype, loader, initial_design_configs, metrics)
    
    Buffer().printflush("AutoML: optimization done.")

    end_time = time.time()
    rules = mine_results(args, buffer, metrics)

    Buffer().printflush("AutoML: miner done.")

    mining_time = time.time()
    dump_results(args, loader, buffer, best_config, rules, start_time, end_time, mining_time)
    
    Buffer().printflush("AutoML: export done.")

    del loader
    del buffer
    del prototype

    return best_config, rules
