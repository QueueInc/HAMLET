import json
from flaml import tune


class Loader:

    _space = {}
    _template_constraints = []
    _instance_constraints = []

    def __init__(self, path=None):
        if path:
            knowledge = _load(path=path)
            _space = _get_space(input_space=knowledge["space"])
            _template_constraints = _get_template_constraints(
                input_template_constraints=knowledge["template_constraints"]
            )
            _instance_constraints = _get_instance_constraints(
                input_instance_constraints=knowledge["instance_constraints"]
            )
        else:
            raise Exception("No knowledge path provided")


def _load(path):
    with open(path) as f:
        data = json.load(f)
    return data


def _get_space(input_space):
    space = {}
    if type(input_space) is not dict:
        return input_space
    for key, value in input_space.items():
        if key == "choice":
            return tune.choice([_get_space(elem) for elem in value])
        if type(value) is dict:
            space[key] = _get_space(value)
        elif type(value) is list:
            raise Exception("You put an array without the 'choice' key")
        else:
            space[key] = value
    return space


def _check_step_assertion(step, operator_config, config):
    if step not in config:
        raise Exception(
            f"The step {step} in a constraint does not exist in the declared search space."
        )
    hyper_parameter_conditions = []
    if step == "Prototype":
        if "neq" in operator_config:
            return config[step] != operator_config["neq"]
        if "eq" in operator_config:
            return config[step] == operator_config["eq"]
        if "gt" in operator_config:
            return config[step] > operator_config["gt"]
        if "ln" in operator_config:
            return config[step] < operator_config["ln"]
        if "gte" in operator_config:
            return config[step] >= operator_config["gte"]
        if "lte" in operator_config:
            return config[step] <= operator_config["lte"]
        raise Exception(
            f"It seems like that in the step {step} you used a comparison operator that is not allowed."
        )
    else:
        for hyper_parameter_key, hyper_parameter_value in operator_config.items():
            if hyper_parameter_key not in config[step]:
                return False

            if "neq" in hyper_parameter_value:
                hyper_parameter_conditions.append(
                    config[step][hyper_parameter_key] != hyper_parameter_value["neq"]
                )
            if "eq" in hyper_parameter_value:
                hyper_parameter_conditions.append(
                    config[step][hyper_parameter_key] == hyper_parameter_value["eq"]
                )
            if "gt" in hyper_parameter_value:
                hyper_parameter_conditions.append(
                    config[step][hyper_parameter_key] > hyper_parameter_value["gt"]
                )
            if "ln" in hyper_parameter_value:
                hyper_parameter_conditions.append(
                    config[step][hyper_parameter_key] < hyper_parameter_value["ln"]
                )
            if "gte" in hyper_parameter_value:
                hyper_parameter_conditions.append(
                    config[step][hyper_parameter_key] >= hyper_parameter_value["gte"]
                )
            if "lte" in hyper_parameter_value:
                hyper_parameter_conditions.append(
                    config[step][hyper_parameter_key] <= hyper_parameter_value["lte"]
                )
        return all(hyper_parameter_conditions)


def _generate_template_constraint(constraint, config):
    for step, operator_config in constraint.items():
        if not _check_step_assertion(step, operator_config, config):
            return False
    return True


def _get_template_constraints(input_template_constraints):
    return [
        lambda config: _generate_template_constraint(constraint, config)
        for constraint in input_template_constraints
    ]


def _get_instance_constraints(input_instance_constraints):
    return input_instance_constraints


def get_space(self):
    return self._space


def get_template_constraints(self):
    return self._template_constraints


def get_instance_constraints(self):
    return self._instance_constraints
