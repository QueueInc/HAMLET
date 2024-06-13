import json
import time
import sys

from ConfigSpace import Configuration
import numpy as np

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from utils.json_to_csv import json_to_csv
from utils.datasets import load_dataset_from_openml
from hamlet.objective import Prototype
from hamlet.loader import Loader
from hamlet.buffer import Buffer
from hamlet.miner import Miner

def run(args):

    np.random.seed(args.seed)
    metrics = [args.fair_metric, args.metric]
    # Put initial initial_design_configs = 0 if not first iteration
    # (brutto tanto ma devo lanciare gli esperimenti :crying_face:)
    initial_design_configs = 5 if args.input_path == "automl_input_1.json" else 0
    X, y, categorical_indicator, sensitive_indicator = load_dataset_from_openml(
        args.dataset
    )
    
    loader = Loader(args.input_path)
    buffer = Buffer(
        metrics=metrics, loader=loader, initial_design_configs=initial_design_configs
    )

    buffer.attach_handler()

    prototype = Prototype(
        X,
        y,
        categorical_indicator,
        sensitive_indicator,
        args.fair_metric,
        args.metric,
        args.mode,
    )

    Buffer().printflush("AutoML: starting optimization.")

    start_time = time.time()
    _, _, best_config = optimize(args, prototype, loader, initial_design_configs, metrics)
    
    Buffer().printflush("AutoML: optimization done.")

    end_time = time.time()
    rules = mine_results(args, buffer)

    Buffer().printflush("AutoML: miner done.")

    mining_time = time.time()
    dump_results(args, loader, buffer, best_config, rules, start_time, end_time, mining_time)
    
    Buffer().printflush("AutoML: export done.")

    del loader
    del buffer
    del prototype


def optimize(args, prototype, loader, initial_design_configs, metrics):

    def _best_configs(prototype, incumbents, incumbents_costs):
        best_config = []
        Buffer().printflush(incumbents)
        try:
            best_config = [
                {
                    **(prototype._transform_configuration(elem.get_dictionary())),
                    **{
                        key: (
                            (
                                float("-inf")
                                if incumbents_costs[idx_incumbent][idx_metric]
                                == float("inf")
                                else (1 - incumbents_costs[idx_incumbent][idx_metric])
                            )
                            if args.mode == "max"
                            else incumbents_costs[idx_incumbent][idx_metric]
                        )
                        for idx_metric, key in enumerate([args.fair_metric, args.metric])
                    },
                }
                for idx_incumbent, elem in enumerate(incumbents)
            ]
        except:
            Buffer().printflush("Apparently no results are available")

        return best_config

    previous_evaluated_points = [
        Configuration(configuration_space=loader.get_space(), values=elem)
        for elem in (
            loader.get_points_to_evaluate(is_smac=True)
            + loader.get_instance_constraints(is_smac=True)
        )
    ]

    # SMAC vuole che specifichiamo i trials, quindi non possiamo mettere -1, va bene maxsize?
    n_trials = (
        (args.batch_size + len(previous_evaluated_points) + initial_design_configs)
        if args.batch_size > 0
        else sys.maxsize
    )

    # Define our environment variables
    scenario = Scenario(
        loader.get_space(),
        objectives=metrics,
        walltime_limit=args.time_budget,
        n_trials=n_trials,
        seed=args.seed,
        n_workers=1,
    )

    initial_design = HPOFacade.get_initial_design(
        scenario,
        n_configs=initial_design_configs,
        additional_configs=previous_evaluated_points,
    )
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        # Questa non funziona di sicuro
        prototype.objective,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    # Let's optimize
    incumbents = smac.optimize()
    incumbents_costs = [smac.runhistory.average_cost(elem) for elem in incumbents]
    return incumbents, incumbents_costs, _best_configs(prototype, incumbents, incumbents_costs)


def mine_results(args, buffer):
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    miners = {
        metric: Miner(
            points_to_evaluate=points_to_evaluate,
            evaluated_rewards=evaluated_rewards,
            metric=args.metric,
            mode=args.mode,
        )
        # for metric in metrics
        for metric in [args.fair_metric]
    }
    return [elem for miner in miners.values() for elem in miner.get_rules()]


def dump_results(args, loader, buffer, best_config, rules, start_time, end_time, mining_time):

    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    graph_generation_time = loader.get_graph_generation_time()
    space_generation_time = loader.get_space_generation_time()

    automl_output = {
        "start_time": start_time,
        "graph_generation_time": graph_generation_time,
        "space_generation_time": space_generation_time,
        "optimization_time": end_time - start_time,
        "mining_time": time.time() - mining_time,
        "best_config": best_config,
        "rules": rules,
        "points_to_evaluate": points_to_evaluate,
        "evaluated_rewards": [
            json.loads(str(reward).replace("'", '"').replace("-inf", '"-inf"'))
            for reward in evaluated_rewards
        ],
    }

    with open(args.output_path, "w") as outfile:
        json.dump(automl_output, outfile)

    json_to_csv(automl_output=automl_output.copy(), args=args)
