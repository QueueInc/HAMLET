import time
import numpy as np

from hamlet.buffer import Buffer
from hamlet.objective import Prototype
from hamlet.loader import Loader
from hamlet.utils.datasets import load_dataset_from_openml
from hamlet.engine import optimize, mine_results, dump_results


def run(args):

    np.random.seed(args.seed)
    loader = Loader(args.input_path)

    settings = loader.get_settings()

    settings["input_path"] = args.input_path
    settings["output_path"] = args.output_path
    settings["seed"] = args.seed

    metrics = [settings["fair_metric"], settings["metric"]]

    (
        X,
        y,
        categorical_indicator,
        sensitive_indicator,
        feature_names,
        encoding_mappings,
    ) = load_dataset_from_openml(settings["dataset"], settings["sensitive_features"])

    initial_design_configs = 5 if len(loader.get_points_to_evaluate()) == 0 else 0

    buffer = Buffer(
        metrics=metrics,
        loader=loader,
        initial_design_configs=initial_design_configs,
    )

    buffer.attach_handler()
    start_time = time.time()

    Buffer().printflush("AutoML: starting optimization.")

    prototype = Prototype(
        X,
        y,
        categorical_indicator,
        sensitive_indicator,
        encoding_mappings,
        feature_names,
        settings["fair_metric"],
        settings["metric"],
        settings["mode"],
    )

    _, _, best_config = optimize(
        settings, prototype, loader, initial_design_configs, metrics
    )

    Buffer().printflush("AutoML: optimization done.")

    end_time = time.time()

    mining_config = {
        settings["fair_metric"]: settings["fairness_thresholds"],
        "by_group": settings["fairness_thresholds"],
        settings["metric"]: settings["performance_thresholds"],
    }
    rules = mine_results(settings, buffer, mining_config, settings["mining_support"])

    Buffer().printflush("AutoML: miner done.")

    mining_time = time.time()
    dump_results(
        settings,
        loader,
        buffer,
        best_config,
        rules,
        start_time,
        end_time,
        mining_time,
        # encoding_mappings,
        metrics,
    )

    Buffer().printflush("AutoML: export done.")

    del loader
    del buffer
    del prototype

    return best_config, rules
