import numpy as np
import pandas as pd

from sequential.seq2pat import Seq2Pat
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

from hamlet.utils import commons


class Miner:

    def __init__(self, points_to_evaluate, evaluated_rewards, metric, mode):
        self._automl_output = [
            (config, reward[metric])
            for config, reward in list(zip(points_to_evaluate, evaluated_rewards))
            if reward["status"] != "previous_constraint"
        ]
        self._min_automl_outputs = 100
        self._metric = metric
        self._mode = mode
        # Pay attention, in this version we assume the metric varies between 0 and 1
        # self.max_reward = max(temp_evaluated_rewards)
        # self.min_reward = min(temp_evaluated_rewards)

    def _clean_prototype(sef, config, classification_flag=False):
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

    def _is_reward_eligible(sef, reward, metric_threshold, mode):
        return (
            reward >= round(metric_threshold, 1)
            if mode == "mandatory"
            else reward <= round(metric_threshold, 1)
        )

    def _get_support_dicts(sef, metric_stat, support_stat, mode):
        return np.arange(
            (
                (metric_stat["max"] - metric_stat["step"])
                if mode == "mandatory"
                else (metric_stat["min"] + metric_stat["step"])
            ),
            (
                (metric_stat["suff"] - metric_stat["step"])
                if mode == "mandatory"
                else ((metric_stat["max"] - metric_stat["suff"]) + metric_stat["step"])
            ),
            -metric_stat["step"] if mode == "mandatory" else metric_stat["step"],
        ), np.arange(
            support_stat["max"] - support_stat["step"],
            support_stat[f"{mode}_suff"] - support_stat["step"],
            -support_stat["step"],
        )

    def _get_presence_rules(self, metric_stat, support_stat, mode):

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

        metric_thresholds, support_thresholds = self._get_support_dicts(
            metric_stat, support_stat, mode
        )

        for metric_threshold in metric_thresholds:
            for support_threshold in support_thresholds:
                for algorithm in commons.algorithms:
                    prototypes = [
                        self._clean_prototype(config, classification_flag=True)
                        for config, reward in self._automl_output
                        if self._is_reward_eligible(reward, metric_threshold, mode)
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
                                    "source": self._metric,
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

    def _mine_order_rules(
        self,
        prototypes,
        metric_threshold,
        support_threshold,
        algorithm,
        mode,
        rules,
        group,
    ):
        current_rules = []
        prototypes = [prototype for prototype in prototypes if algorithm in prototype]
        if len(prototypes) > self._min_automl_outputs:
            seq2pat = Seq2Pat(sequences=prototypes)
            support = int(round(support_threshold, 1) * len(prototypes))
            if support > 0:
                mined_rules = seq2pat.get_patterns(min_frequency=support)
                if len(mined_rules) > 0:
                    mined_rules = [
                        {
                            "source": self._metric,
                            "type": f"{mode}_order",
                            "rule": rule[:-1],
                            "support": round(rule[-1] / len(prototypes), 2),
                            "occurrences": rule[-1],
                            "considered_configurations": len(prototypes),
                            "metric_threshold": round(metric_threshold, 1),
                            "group": group,
                        }
                        for rule in mined_rules
                    ]
                    mined_rules = [
                        mined_rule
                        for mined_rule in mined_rules
                        if (mined_rule["rule"] not in [rule["rule"] for rule in rules])
                        and (len(mined_rule["rule"]) == 3)
                        and (mined_rule["rule"][2] in commons.algorithms)
                    ]
                    current_rules += mined_rules
        return current_rules

    def _get_order_rules(self, metric_stat, support_stat, mode, by_group):

        rules = []

        metric_thresholds, support_thresholds = self._get_support_dicts(
            metric_stat, support_stat, mode
        )

        for metric_threshold in metric_thresholds:
            for support_threshold in support_thresholds:
                for algorithm in commons.algorithms:
                    if by_group:
                        for group in self._automl_output[0][1].keys():
                            prototypes = [
                                self._clean_prototype(config, classification_flag=True)
                                for config, reward in self._automl_output
                                if self._is_reward_eligible(
                                    reward[group], metric_threshold, mode
                                )
                            ]
                            rules += self._mine_order_rules(
                                prototypes,
                                metric_threshold,
                                support_threshold,
                                algorithm,
                                mode,
                                rules,
                                group,
                            )
                    else:
                        prototypes = [
                            self._clean_prototype(config, classification_flag=True)
                            for config, reward in self._automl_output
                            if self._is_reward_eligible(reward, metric_threshold, mode)
                        ]
                        rules += self._mine_order_rules(
                            prototypes,
                            metric_threshold,
                            support_threshold,
                            algorithm,
                            mode,
                            rules,
                            "None",
                        )
        if by_group:
            return rules
        else:
            return [
                new_rule
                for new_rule in rules
                if [new_rule["rule"][1], new_rule["rule"][0], new_rule["rule"][2]]
                not in [rule["rule"] for rule in rules]
            ]

    def get_rules(self):
        rules = []
        metric_stat = {"min": 0, "max": 1, "step": 0.1, "suff": 0.8}
        support_stat = {
            "min": 0,
            "max": 1,
            "step": 0.1,
            "mandatory_suff": 0.8,
            "forbidden_suff": 0.8,
        }
        rules += self._get_order_rules(
            metric_stat=metric_stat,
            support_stat=support_stat,
            mode="forbidden" if self._metric == "by_group" else "mandatory",
            by_group=self._metric == "by_group",
        )
        if self._metric != "by_group":
            mandatory_rules = self._get_presence_rules(
                metric_stat=metric_stat, support_stat=support_stat, mode="mandatory"
            )
            rules += mandatory_rules
            rules += [
                forbidden_rule
                for forbidden_rule in self._get_presence_rules(
                    metric_stat=metric_stat, support_stat=support_stat, mode="forbidden"
                )
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
