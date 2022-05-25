from math import floor
import numpy as np
from lzma import MODE_FAST
from sequential.seq2pat import Seq2Pat


class Miner:
    def __init__(self, points_to_evaluate, evaluated_rewards, metric, mode):
        temp_evaluated_rewards = [elem[metric] for elem in evaluated_rewards]
        self._valid_automl_outputs = len(
            [elem for elem in points_to_evaluate if elem != float("-inf")]
        )
        self._min_automl_outputs = int(self._valid_automl_outputs / 5)
        self._automl_output = list(zip(points_to_evaluate, temp_evaluated_rewards))
        self._metric = metric
        self._mode = mode
        # Pay attention, in this version we assume the metric varies between 0 and 1
        # self.max_reward = max(temp_evaluated_rewards)
        # self.min_reward = min(temp_evaluated_rewards)
        self._metric_stat = {"min": 0, "max": 1, "step": 0.1, "suff": 0.6}
        self._support_stat = {"min": 0, "max": 1, "step": 0.1, "suff": 0.1}

    def _get_prototype_rules(self):
        def clean_prototype(config):
            prototype = config["prototype"].split("_")
            true_steps = [
                key
                for key, value in config.items()
                if key != "prototype" and value["type"] != "FunctionTransformer"
            ]
            return [elem for elem in prototype if elem in true_steps]

        rules = []
        for metric_threshold in np.arange(
            self._metric_stat["min"] + self._metric_stat["suff"],
            self._metric_stat["max"],
            self._metric_stat["step"],
        ):
            for support_threshold in np.arange(
                self._support_stat["min"] + self._support_stat["suff"],
                self._support_stat["max"],
                self._support_stat["step"],
            ):
                prototypes = [
                    clean_prototype(config)
                    for config, reward in self._automl_output
                    if reward >= round(metric_threshold, 1)
                ]
                prototypes = [
                    prototype[:-1] for prototype in prototypes if len(prototype) >= 2
                ]
                if len(prototypes) > self._min_automl_outputs:
                    seq2pat = Seq2Pat(sequences=prototypes)
                    support = int(support_threshold * len(prototypes))
                    current_rules = seq2pat.get_patterns(min_frequency=support)
                    if len(current_rules) > 0:
                        rules.append((current_rules, round(metric_threshold, 1)))
        return rules

    def get_rules(self):
        rules = []
        rules += self._get_prototype_rules()
        return rules
