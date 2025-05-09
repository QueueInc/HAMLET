import json

from hamlet.utils.flaml_to_smac import get_space, flatten_configuration


class Loader:

    _space = {}
    _template_constraints = []
    _instance_constraints = []
    _points_to_evaluate = []
    _evaluated_rewards = []
    _settings = {}

    def __init__(self, path=None):

        if not path:
            raise Exception("No knowledge path provided")

        knowledge = self._load(path=path)

        self._settings = {
            x : knowledge[x] for x in ["sensitive_features", "metric", "fair_metric", "dataset", "mode", "batch_size", "time_budget", "fairness_thresholds", "performance_thresholds", "mining_support"]
        }

        self._graph_generation_time = knowledge["graph_generation_time"]
        self._space_generation_time = knowledge["space_generation_time"]

        for constraint in knowledge["template_constraints"]:
            self._template_constraints.append(self._get_template_constraint(constraint))

        self._evaluated_rewards = self._get_evaluated_rewards(
            input_evaluated_rewards=knowledge["evaluated_rewards"]
        )

        self._space, self._instance_constraints, self._points_to_evaluate = get_space(
            knowledge
        )

    def _load(self, path):
        with open(path) as f:
            data = json.load(f)
        return data

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
                flatten_configuration(config) for config in self._instance_constraints
            ]
        else:
            return self._instance_constraints

    def get_points_to_evaluate(self, is_smac=False):
        if is_smac:
            return [
                flatten_configuration(config) for config in self._points_to_evaluate
            ]
        else:
            return self._points_to_evaluate

    def get_evaluated_rewards(self):
        return self._evaluated_rewards

    def get_settings(self):
        return self._settings

