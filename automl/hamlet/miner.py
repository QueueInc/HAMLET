import itertools
import numpy as np
import pandas as pd

from math import floor
from lzma import MODE_FAST
from sequential.seq2pat import Seq2Pat
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from utils import commons


class Miner:
    def __init__(self, points_to_evaluate, evaluated_rewards, metric, mode):
        self._automl_output = [
            (config, reward[metric])
            for config, reward in list(zip(points_to_evaluate, evaluated_rewards))
            if reward["status"] != "previous_constraint"
        ]
        self._min_automl_outputs = int(len(self._automl_output) / 5)
        self._metric = metric
        self._mode = mode
        # Pay attention, in this version we assume the metric varies between 0 and 1
        # self.max_reward = max(temp_evaluated_rewards)
        # self.min_reward = min(temp_evaluated_rewards)
        self._metric_stat = {"min": 0, "max": 1, "step": 0.1, "suff": 0.6}
        self._support_stat = {"min": 0, "max": 1, "step": 0.1, "suff": 0.5}

    def _clean_prototype(config, classification_flag=False):
        prototype = config["prototype"].split("_")
        true_steps = [
            key
            for key, value in config.items()
            if key != "prototype" and value["type"] != "FunctionTransformer"
        ]
        clean_prototype = [elem for elem in prototype if elem in true_steps]
        if classification_flag:
            clean_prototype[-1] = config["classification"]["type"]
        return clean_prototype

    def _get_presence_rules(self, mode):
        def is_reward_eligible(reward):
            return (
                reward >= round(metric_threshold, 1)
                if mode == "mandatory"
                else reward <= round(metric_threshold, 1)
            )

        def maximal_elements(rules):
            return [
                new_rule
                for new_rule in rules
                if any([elem in commons.algorithms for elem in new_rule["rule"]])
                and not (
                    any(
                        [
                            all(elem in rule["rule"] for elem in new_rule["rule"])
                            for rule in rules
                            if rule["rule"] != new_rule["rule"]
                        ]
                    )
                )
            ]

        def minimal_elements(rules):
            return [
                new_rule
                for new_rule in rules
                if any([elem in commons.algorithms for elem in new_rule["rule"]])
                and len(new_rule["rule"]) <= 2
            ]

        rules = []

        metric_thresholds = np.arange(
            self._metric_stat["max"] - self._metric_stat["step"]
            if mode == "mandatory"
            else self._metric_stat["min"] + self._metric_stat["step"],
            self._metric_stat["suff"] - self._metric_stat["step"]
            if mode == "mandatory"
            else (self._metric_stat["max"] - self._metric_stat["suff"])
            + self._metric_stat["step"],
            -self._metric_stat["step"]
            if mode == "mandatory"
            else self._metric_stat["step"],
        )
        support_thresholds = np.arange(
            self._support_stat["max"] - self._support_stat["step"],
            self._support_stat["suff"] - self._support_stat["step"],
            -self._support_stat["step"],
        )
        for metric_threshold in metric_thresholds:
            for support_threshold in support_thresholds:
                for algorithm in commons.algorithms:
                    prototypes = [
                        Miner._clean_prototype(config, classification_flag=True)
                        for config, reward in self._automl_output
                        if is_reward_eligible(reward)
                    ]
                    prototypes = [
                        prototype for prototype in prototypes if algorithm in prototype
                    ]
                    if len(prototypes) > self._min_automl_outputs:
                        tr = TransactionEncoder()
                        tr_arr = tr.fit_transform(prototypes)
                        df = pd.DataFrame(tr_arr, columns=tr.columns_)
                        frequent_itemsets = apriori(
                            df,
                            min_support=round(support_threshold, 1),
                            use_colnames=True,
                        )
                        if frequent_itemsets.shape[0] > 0:
                            current_rules = [
                                {
                                    "type": mode,
                                    "rule": list(rule["itemsets"]),
                                    "support": round(rule["support"], 2),
                                    "occurrences": int(
                                        rule["support"] * len(prototypes)
                                    ),
                                    "considered_configurations": len(prototypes),
                                    "metric_threshold": round(metric_threshold, 1),
                                }
                                for index, rule in frequent_itemsets.to_dict(
                                    "index"
                                ).items()
                            ]
                            current_rules = [
                                current_rule
                                for current_rule in current_rules
                                if current_rule["rule"]
                                not in [rule["rule"] for rule in rules]
                            ]
                            rules += current_rules
        return maximal_elements(rules)

    def _get_order_rules(self):
        rules = []
        metric_thresholds = np.arange(
            self._metric_stat["max"] - self._metric_stat["step"],
            self._metric_stat["suff"] - self._metric_stat["step"],
            -self._metric_stat["step"],
        )
        support_thresholds = np.arange(
            self._support_stat["max"] - self._support_stat["step"],
            self._support_stat["suff"] - self._support_stat["step"],
            -self._support_stat["step"],
        )
        for metric_threshold in metric_thresholds:
            for support_threshold in support_thresholds:
                for algorithm in commons.algorithms:
                    prototypes = [
                        Miner._clean_prototype(config, classification_flag=True)
                        for config, reward in self._automl_output
                        if reward >= round(metric_threshold, 1)
                    ]
                    # prototypes = [
                    #     prototype[:-1] for prototype in prototypes if len(prototype) >= 2
                    # ]
                    prototypes = [
                        prototype for prototype in prototypes if algorithm in prototype
                    ]
                    if len(prototypes) > self._min_automl_outputs:
                        seq2pat = Seq2Pat(sequences=prototypes)
                        support = int(round(support_threshold, 1) * len(prototypes))
                        if support > 0:
                            current_rules = seq2pat.get_patterns(min_frequency=support)
                            if len(current_rules) > 0:
                                current_rules = [
                                    {
                                        "type": "mandatory_order",
                                        "rule": rule[:-1],
                                        "support": round(rule[-1] / len(prototypes), 2),
                                        "occurrences": rule[-1],
                                        "considered_configurations": len(prototypes),
                                        "metric_threshold": round(metric_threshold, 1),
                                    }
                                    for rule in current_rules
                                ]
                                current_rules = [
                                    current_rule
                                    for current_rule in current_rules
                                    if (
                                        current_rule["rule"]
                                        not in [rule["rule"] for rule in rules]
                                    )
                                    and (len(current_rule["rule"]) == 3)
                                    and (current_rule["rule"][2] in commons.algorithms)
                                ]
                                rules += current_rules
        return [
            new_rule
            for new_rule in rules
            if [new_rule["rule"][1], new_rule["rule"][0], new_rule["rule"][2]]
            not in [rule["rule"] for rule in rules]
        ]

    def get_rules(self):
        rules = []
        rules += self._get_order_rules()
        mandatory_rules = self._get_presence_rules(mode="mandatory")
        rules += mandatory_rules
        rules += [
            forbidden_rule
            for forbidden_rule in self._get_presence_rules(mode="forbidden")
            if not any(
                [
                    all(
                        elem in mandatory_rule["rule"]
                        for elem in forbidden_rule["rule"]
                    )
                    for mandatory_rule in mandatory_rules
                ]
            )
        ]
        return rules
