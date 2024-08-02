import json
import time
import sys

from ConfigSpace import Configuration

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from hamlet.buffer import Buffer
from hamlet.miner import Miner

from hamlet.utils.json_to_csv import json_to_csv
from hamlet.utils.flaml_to_smac import flatten_configuration, transform_configuration


def optimize(args, prototype, loader, initial_design_configs, metrics):

    def _best_configs(incumbents, incumbents_costs):
        best_config = []
        try:
            best_config = [
                {
                    **(transform_configuration(elem.get_dictionary())),
                    **{
                        key: (
                            (
                                '"-inf"'
                                if incumbents_costs[idx_incumbent][idx_metric]
                                == float("inf")
                                else (1 - incumbents_costs[idx_incumbent][idx_metric])
                            )
                            if args.mode == "max"
                            else incumbents_costs[idx_incumbent][idx_metric]
                        )
                        for idx_metric, key in enumerate(
                            [args.fair_metric, args.metric]
                        )
                    },
                }
                for idx_incumbent, elem in enumerate(incumbents)
            ]
        except:
            Buffer().printflush("Apparently no results are available")

        return best_config

    _configs, _ = Buffer()._filter_previous_results(
        loader.get_points_to_evaluate(),
        loader.get_evaluated_rewards(),
        metrics,
    )
    previous_evaluated_points = [
        Configuration(configuration_space=loader.get_space(), values=elem)
        for elem in (
            [flatten_configuration(config) for config in _configs]
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
        # trial_walltime_limit=900
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
        logging_level=40,
    )

    # Let's optimize
    incumbents = smac.optimize()
    incumbents_costs = [smac.runhistory.average_cost(elem) for elem in incumbents]
    return incumbents, incumbents_costs, _best_configs(incumbents, incumbents_costs)


def mine_results(args, buffer, metrics):
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    miners = {
        m: Miner(
            points_to_evaluate=points_to_evaluate,
            evaluated_rewards=evaluated_rewards,
            metric=m,
            mode=args.mode,
        )
        for m in metrics
    }
    return [elem for miner in miners.values() for elem in miner.get_rules()]


def dump_results(
    args, loader, buffer, best_config, rules, start_time, end_time, mining_time
):

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
