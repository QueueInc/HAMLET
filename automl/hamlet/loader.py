import json

import copy

from flaml import tune
from flaml import CFO
from flaml.tune.space import complete_config

import ConfigSpace

from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    EqualsCondition,
    Configuration,
)


class Loader:

    _space = {}
    _template_constraints = []
    _instance_constraints = []
    _points_to_evaluate = []
    _evaluated_rewards = []

    def __init__(self, path=None):
        if path:
            knowledge = self._load(path=path)
            self._graph_generation_time = knowledge["graph_generation_time"]
            self._space_generation_time = knowledge["space_generation_time"]

            ##################### IMPORT CON CONVERSIONE A FLAML
            flaml_space = self._get_space(input_space=knowledge["space"])
            for constraint in knowledge["template_constraints"]:
                self._template_constraints.append(
                    self._get_template_constraint(constraint)
                )
            flaml_instance_constraints = self._get_instance_constraints(
                input_instance_constraints=knowledge["instance_constraints"]
            )
            flaml_points_to_evaluate = self._get_points_to_evaluate(
                input_points_to_evaluate=knowledge["points_to_evaluate"]
            )
            self._evaluated_rewards = self._get_evaluated_rewards(
                input_evaluated_rewards=knowledge["evaluated_rewards"]
            )

            ##################### SMAC MAPPING

            # Complete instamce constraints
            cfo = CFO(
                space=flaml_space,
                metric="metric",
                mode="max",
            )

            self._instance_constraints = [
                complete_config(config, flaml_space, cfo._ls)[0]
                for config in flaml_instance_constraints
            ]

            self._points_to_evaluate = [
                complete_config(config, flaml_space, cfo._ls)[0]
                for config in flaml_points_to_evaluate
            ]

            # Map space to smac space
            self._space = self._space_to_configspace(knowledge["space"])
        else:
            raise Exception("No knowledge path provided")

    ##################### SMAC MAPPING

    def _create_hyperparameter(self, name, definition):
        if "choice" in definition and all(
            not isinstance(i, dict) for i in definition["choice"]
        ):
            return CategoricalHyperparameter(name, definition["choice"])
        elif "randint" in definition:
            return UniformIntegerHyperparameter(
                name, lower=definition["randint"][0], upper=definition["randint"][1]
            )
        elif "uniform" in definition:
            return UniformFloatHyperparameter(
                name, lower=definition["uniform"][0], upper=definition["uniform"][1]
            )
        else:
            raise ValueError(f"Unknown hyperparameter definition: {definition}")

    def _flatten_configuration(self, config):
        flattened = {}

        def recurse(current_level, prefix):
            print(current_level)
            for key, value in current_level.items():
                new_key = (
                    f"""{prefix}.{current_level["type"]}.{key}"""
                    if key != "type"
                    else prefix
                )
                flattened[new_key] = value

        for key, value in config.items():
            if isinstance(value, dict) and "type" in value:
                flattened[key] = value["type"]
                recurse(value, key)
            else:
                flattened[key] = value

        # to_return = copy.deepcopy(flattened)
        # for key, value in flattened.items():
        #     if "type" in key:
        #         del to_return[key]

        return flattened

    def _add_hyperparameters(
        self, cs, name_prefix, params, conditions, parent_name=None, parent_value=None
    ):
        for key, value in params.items():
            full_name = f"{name_prefix}.{key}" if name_prefix else key
            if isinstance(value, dict) and "type" in value:
                sub_name_prefix = f"{full_name}.{value['type']}"
                for sub_key, sub_value in value.items():
                    if sub_key != "type":
                        self._add_hyperparameters(
                            cs,
                            sub_name_prefix,
                            {sub_key: sub_value},
                            conditions,
                            parent_name=full_name,
                            parent_value=value["type"],
                        )
            elif (
                isinstance(value, dict)
                and "choice" in value
                and all(isinstance(i, dict) for i in value["choice"])
            ):
                choices = [i["type"] for i in value["choice"]]
                hp = CategoricalHyperparameter(full_name, choices)
                cs.add_hyperparameter(hp)
                for choice_dict in value["choice"]:
                    choice_type = choice_dict["type"]
                    sub_name_prefix = f"{full_name}.{choice_type}"
                    for sub_key, sub_value in choice_dict.items():
                        if sub_key != "type":
                            self._add_hyperparameters(
                                cs,
                                sub_name_prefix,
                                {sub_key: sub_value},
                                conditions,
                                parent_name=full_name,
                                parent_value=choice_type,
                            )
            else:
                hp = self._create_hyperparameter(full_name, value)
                cs.add_hyperparameter(hp)
                if parent_name:
                    condition = EqualsCondition(
                        hp, cs.get_hyperparameter(parent_name), parent_value
                    )
                    conditions.append(condition)

    def _space_to_configspace(self, space):
        cs = ConfigurationSpace()
        conditions = []

        for top_level_key, top_level_value in space.items():
            if "choice" in top_level_value and all(
                not isinstance(i, dict) for i in top_level_value["choice"]
            ):
                hp = CategoricalHyperparameter(top_level_key, top_level_value["choice"])
                cs.add_hyperparameter(hp)
            elif "choice" in top_level_value and all(
                isinstance(i, dict) for i in top_level_value["choice"]
            ):
                choices = [i["type"] for i in top_level_value["choice"]]
                hp = CategoricalHyperparameter(top_level_key, choices)
                cs.add_hyperparameter(hp)
                for choice_dict in top_level_value["choice"]:
                    choice_type = choice_dict["type"]
                    sub_name_prefix = f"{top_level_key}.{choice_type}"
                    for sub_key, sub_value in choice_dict.items():
                        if sub_key != "type":
                            self._add_hyperparameters(
                                cs,
                                sub_name_prefix,
                                {sub_key: sub_value},
                                conditions,
                                parent_name=top_level_key,
                                parent_value=choice_type,
                            )
            else:
                raise ValueError(
                    f"Top-level key {top_level_key} must have a 'choice' key"
                )

        cs.add_conditions(conditions)
        return cs

    def _load(self, path):
        with open(path) as f:
            data = json.load(f)
        return data

    def _get_space(self, input_space):
        space = {}
        if type(input_space) is not dict:
            return input_space
        for key, value in input_space.items():
            if key == "choice":
                return tune.choice([self._get_space(elem) for elem in value])
            if key == "randint":
                return tune.randint(lower=value[0], upper=value[1])
            if key == "uniform":
                return tune.uniform(lower=value[0], upper=value[1])
            if type(value) is dict:
                space[key] = self._get_space(value)
            elif type(value) is list:
                raise Exception("You put an array without the 'choice' key")
            else:
                space[key] = value
        return space

    def _check(self, step, value_constraint, target_value):
        if "neq" in value_constraint:
            return target_value != value_constraint["neq"]
        if "eq" in value_constraint:
            return target_value == value_constraint["eq"]
        if "gt" in value_constraint:
            return target_value > value_constraint["gt"]
        if "ln" in value_constraint:
            return target_value < value_constraint["ln"]
        if "gte" in value_constraint:
            return target_value >= value_constraint["gte"]
        if "lte" in value_constraint:
            return target_value <= value_constraint["lte"]
        if "in" in value_constraint:
            return target_value in value_constraint["in"]
        if "nin" in value_constraint:
            return target_value not in value_constraint["nin"]
        raise Exception(
            f"It seems like that in the step {step} you used a comparison operator that is not allowed."
        )

    def _check_step_assertion(self, step, operator_config, config):
        if step not in config:
            raise Exception(
                f"The step {step} in a constraint does not exist in the declared search space."
            )

        if step == "prototype":
            return self._check(step, operator_config, config[step])

        hyper_parameter_conditions = []
        for hyper_parameter_key, hyper_parameter_value in operator_config.items():
            if hyper_parameter_key not in config[step]:
                return False
            hyper_parameter_conditions.append(
                self._check(
                    step, hyper_parameter_value, config[step][hyper_parameter_key]
                )
            )
        return all(hyper_parameter_conditions)

    def _generate_template_constraint(self, constraint, config):
        for step, operator_config in constraint.items():
            if not self._check_step_assertion(step, operator_config, config):
                return False
        return True

    def _get_template_constraint(self, constraint):
        return lambda config: self._generate_template_constraint(constraint, config)

    def _get_instance_constraints(self, input_instance_constraints):
        return input_instance_constraints

    def _get_points_to_evaluate(self, input_points_to_evaluate):
        return input_points_to_evaluate

    def _get_evaluated_rewards(self, input_evaluated_rewards):
        return input_evaluated_rewards

    def get_space(self):
        return self._space

    def get_graph_generation_time(self):
        return self._graph_generation_time

    def get_space_generation_time(self):
        return self._space_generation_time

    def get_template_constraints(self):
        return self._template_constraints

    def get_instance_constraints(self, is_smac=False):
        if is_smac:
            return [
                self._flatten_configuration(config)
                for config in self._instance_constraints
            ]
        else:
            return self._instance_constraints

    def get_points_to_evaluate(self, is_smac=False):
        if is_smac:
            return [
                self._flatten_configuration(config)
                for config in self._points_to_evaluate
            ]
        else:
            return self._points_to_evaluate

    def get_evaluated_rewards(self):
        return self._evaluated_rewards
