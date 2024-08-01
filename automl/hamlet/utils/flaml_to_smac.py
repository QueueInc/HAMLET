import copy

import ConfigSpace

from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    EqualsCondition,
    Configuration,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    ForbiddenAndConjunction,
)

from flaml import tune
from flaml import CFO
from flaml.tune.space import complete_config

##################### SMAC TO FLAML


def transform_configuration(config):
    transformed = {}

    for key, value in config.items():
        parts = key.split(".")

        if len(parts) == 1:
            transformed[parts[0]] = value
        else:
            current_level = transformed
            for part in parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                elif isinstance(current_level[part], str):
                    current_level[part] = {"type": current_level[part]}
                current_level = current_level[part]
            current_level[parts[-1]] = value

    for top_level_key in list(transformed.keys()):
        if isinstance(transformed[top_level_key], str):
            type_value = transformed.pop(top_level_key)
            transformed[top_level_key] = {"type": type_value}
        elif (
            isinstance(transformed[top_level_key], dict)
            and "type" not in transformed[top_level_key]
        ):
            type_value = transformed[top_level_key].pop("type", None)
            if type_value:
                transformed[top_level_key] = {
                    "type": type_value,
                    **transformed[top_level_key],
                }

    temp = copy.deepcopy(transformed)
    for key, value in transformed.items():
        if type(value) == dict:
            for nested_key, nested_value in value.items():
                if type(nested_value) == dict:
                    for new_key, new_value in nested_value.items():
                        temp[key][new_key] = new_value
                    del temp[key][nested_key]
    temp["prototype"] = temp["prototype"]["type"]

    return temp


def transform_result(result, metric, fair_metric, mode):
    return {
        key: (
            (float("inf") if result[key] == float("-inf") else (1 - result[key]))
            if mode == "max"
            else result[key]
        )
        for key in [metric, fair_metric]
    }


##################### FLAML TO SMAC


def get_space(knowledge):

    flaml_space = _get_flaml_space(input_space=knowledge["space"])
    flaml_instance_constraints = knowledge["instance_constraints"]
    flaml_points_to_evaluate = knowledge["points_to_evaluate"]

    # Complete instance constraints
    cfo = CFO(
        space=flaml_space,
        metric="metric",
        mode="max",
    )

    instance_constraints = [
        complete_config(config, flaml_space, cfo._ls)[0]
        for config in flaml_instance_constraints
    ]

    # points_to_evaluate = [
    #     complete_config(config, flaml_space, cfo._ls)[0]
    #     for config in flaml_points_to_evaluate
    # ]

    space = _space_to_configspace(knowledge["space"], knowledge["template_constraints"])

    return space, instance_constraints, flaml_points_to_evaluate


def flatten_configuration(config):
    flattened = {}

    def recurse(current_level, prefix):
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


def _get_flaml_space(input_space):
    space = {}
    if type(input_space) is not dict:
        return input_space
    for key, value in input_space.items():
        if key == "choice":
            return tune.choice([_get_flaml_space(elem) for elem in value])
        if key == "randint":
            return tune.randint(lower=value[0], upper=value[1])
        if key == "uniform":
            return tune.uniform(lower=value[0], upper=value[1])
        if type(value) is dict:
            space[key] = _get_flaml_space(value)
        elif type(value) is list:
            raise Exception("You put an array without the 'choice' key")
        else:
            space[key] = value
    return space


def _create_hyperparameter(name, definition):
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


def _add_hyperparameters(
    cs, name_prefix, params, conditions, parent_name=None, parent_value=None
):
    for key, value in params.items():
        full_name = f"{name_prefix}.{key}" if name_prefix else key
        if isinstance(value, dict) and "type" in value:
            sub_name_prefix = f"{full_name}.{value['type']}"
            for sub_key, sub_value in value.items():
                if sub_key != "type":
                    _add_hyperparameters(
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
                        _add_hyperparameters(
                            cs,
                            sub_name_prefix,
                            {sub_key: sub_value},
                            conditions,
                            parent_name=full_name,
                            parent_value=choice_type,
                        )
        else:
            hp = _create_hyperparameter(full_name, value)
            cs.add_hyperparameter(hp)
            if parent_name:
                condition = EqualsCondition(hp, cs[parent_name], parent_value)
                conditions.append(condition)


def _convert_constraints_to_forbidden_clauses(cs, constraints):
    forbidden_clauses = []

    for constraint in constraints:
        clause_elements = []
        for comp_name, conditions in constraint.items():
            for key, value in conditions.items():
                hp_space = cs[comp_name].choices
                if key == "type":
                    cond_value = value
                else:
                    cond_value = {key: value}
                if "eq" in cond_value:
                    clause_elements.append(
                        ForbiddenEqualsClause(cs[comp_name], cond_value["eq"])
                    )
                elif "neq" in cond_value:
                    remaining_values = [
                        val for val in hp_space if val != cond_value["neq"]
                    ]
                    if len(remaining_values) == 1:
                        clause_elements.append(
                            ForbiddenEqualsClause(cs[comp_name], remaining_values[0])
                        )
                    else:
                        clause_elements.append(
                            ForbiddenInClause(cs[comp_name], remaining_values)
                        )
                elif "in" in cond_value:
                    clause_elements.append(
                        ForbiddenInClause(cs[comp_name], cond_value["in"])
                    )
                elif "nin" in cond_value:
                    remaining_values = [
                        val for val in hp_space if val not in cond_value["nin"]
                    ]
                    if len(remaining_values) == 1:
                        clause_elements.append(
                            ForbiddenEqualsClause(cs[comp_name], remaining_values[0])
                        )
                    else:
                        clause_elements.append(
                            ForbiddenInClause(cs[comp_name], remaining_values)
                        )

        if clause_elements:
            forbidden_clauses.append(ForbiddenAndConjunction(*clause_elements))

    return forbidden_clauses


def _space_to_configspace(space, constraints):
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
                        _add_hyperparameters(
                            cs,
                            sub_name_prefix,
                            {sub_key: sub_value},
                            conditions,
                            parent_name=top_level_key,
                            parent_value=choice_type,
                        )
        else:
            raise ValueError(f"Top-level key {top_level_key} must have a 'choice' key")

    cs.add_conditions(conditions)

    # Add forbidden clauses based on constraints
    forbidden_clauses = _convert_constraints_to_forbidden_clauses(cs, constraints)
    for clause in forbidden_clauses:
        try:
            print("Trying to add:", clause)
            cs.add_forbidden_clause(clause)
            print("Added successfully:", clause)
        except Exception as e:
            print("Failed to add:", clause)
            print("Reason:", e)

    return cs
