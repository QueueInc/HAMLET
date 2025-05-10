import json
import time
import sys
import re

import numpy as np

from ConfigSpace import Configuration

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from hamlet.buffer import Buffer
from hamlet.miner import Miner

from hamlet.utils.json_to_csv import json_to_csv
from hamlet.utils.flaml_to_smac import flatten_configuration, transform_configuration

from hamlet.utils.numpyencoder import NumpyEncoder


def optimize(settings, prototype, loader, initial_design_configs, metrics):

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
                            if settings["mode"] == "max"
                            else incumbents_costs[idx_incumbent][idx_metric]
                        )
                        for idx_metric, key in enumerate(
                            [settings["fair_metric"], settings["metric"]]
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
        (
            settings["batch_size"]
            + len(previous_evaluated_points)
            + initial_design_configs
        )
        if settings["batch_size"] > 0
        else sys.maxsize
    )

    # Define our environment variables
    scenario = Scenario(
        loader.get_space(),
        objectives=metrics,
        walltime_limit=settings["time_budget"],
        n_trials=n_trials,
        seed=settings["seed"],
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


def mine_results(settings, buffer, metrics, support):
    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    miners = {
        m: Miner(
            points_to_evaluate=points_to_evaluate,
            evaluated_rewards=evaluated_rewards,
            metric=m,
            mode=settings["mode"],
            support=support,
            thresholds=t,
        )
        for m, t in metrics.items()
    }
    return [elem for miner in miners.values() for elem in miner.get_rules()]


def dump_results(
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
):

    points_to_evaluate, evaluated_rewards = buffer.get_evaluations()
    graph_generation_time = loader.get_graph_generation_time()
    space_generation_time = loader.get_space_generation_time()

    stringify_invalid = lambda x: (
        "nan"
        if np.isnan(x)
        else ("-inf" if x == float("-inf") else ("inf" if x == float("inf") else x))
    )

    # TO ADD IF WE KEEP by_group WITH THE RAW FINE-GRAINED VALUES OF EACH FOLD
    # support_mapping = [
    #     encoding_mappings[sens_feat] for sens_feat in sorted(encoding_mappings)
    # ]

    for reward in evaluated_rewards:
        # TO ADD IF WE KEEP by_group WITH THE RAW FINE-GRAINED VALUES OF EACH FOLD
        # reward["by_group"] = {
        #     "_".join(
        #         [
        #             support_mapping[sens_feat][int(sens_group)]
        #             for sens_feat, sens_group in enumerate(key)
        #         ]
        #     ): "_".join([str(v) for v in value])
        #     for key, value in reward["by_group"].items()
        # }
        for metric in metrics:
            reward[metric] = stringify_invalid(reward[metric])
        reward["by_group"] = {
            key: stringify_invalid(value) for key, value in reward["by_group"].items()
        }

    automl_output = {
        "start_time": start_time,
        "graph_generation_time": graph_generation_time,
        "space_generation_time": space_generation_time,
        "optimization_time": end_time - start_time,
        "mining_time": time.time() - mining_time,
        "best_config": best_config,
        "rules": rules,
        "points_to_evaluate": points_to_evaluate,
        "evaluated_rewards": evaluated_rewards,
        # [
        #     json.loads(str(reward).replace("'", '"').replace("-inf", '"-inf"')).replace(
        #         "'", '"'
        #     )
        #     for reward in evaluated_rewards
        # ],
    }

    with open(settings["output_path"], "w") as outfile:
        json.dump(automl_output, outfile, cls=NumpyEncoder)

    json_to_csv(automl_output=automl_output.copy(), settings=settings)
