from flaml import tune

# Mandatory
# [s1, s2, s3][[op1 V op2] A [op3 V op4] A [op5 V op6]], algorithm)
# {
#     "s1": {"type": {"nin": ["op1", "op2"]}},
#     ...
#     "classification": {"type": {"eq": "algorithm"}}
# }

# Forbidden
# [s1, s2, s3][[op1 V op2] A [op3 V op4] A [op5 V op6]], algorithm)
# {
#     "s1": {"type": {"in": ["op1", "op2"]}},
#     ...
#     "classification": {"type": {"eq": "algorithm"}}
# }

# hyperparameter_exception(classification, dt, max_depth, gt, 2)
# {
#     "classification": {
#         "type": {"eq": "dt"},
#         "max_depth": {"gt": 1},
#     },
# }


prolog_output = {
    "space": {
        "Normalization": {
            "choice": [
                {
                    "type": "FunctionTransformer",
                },
                {
                    "type": "StandardScaler",
                    "with_mean": {"choice": [True, False]},
                    "with_std": {"choice": [True, False]},
                },
                {
                    "type": "MinMaxScaler",
                },
            ]
        },
        "FeatureEngineering": {
            "choice": [
                {
                    "type": "FunctionTransformer",
                },
                {
                    "type": "SelectKBest",
                    "n_features": {"randint": (1, 10)},
                },
                {
                    "type": "PCA",
                    "n_features": {"randint": (1, 10)},
                },
            ]
        },
        "Classification": {
            "choice": [
                {
                    "type": "KNeighborsClassifier",
                    "n_neighbors": {"choice": [3, 5, 7, 9, 11, 13, 15, 17, 19]},
                }
            ]
        },
        "Prototype": {
            "choice": [
                "FeaturesEngineering_Normalization_Classification",
                "Normalization_FeaturesEngineering_Classification",
            ]
        },
    },
    "template_constraints": [
        # if
        #   Normalization is not instantiated with FunctionTransformer and
        #   Discretization is not instantiated with FunctionTransformer
        # {
        #     "Normalization": {"type": {"neq": "FunctionTransformer"}},
        #     "Discretization": {"type": {"neq": "FunctionTransformer"}},
        # },
        # if
        #   DecisionTreeClassifier is instantiated with DecisionTreeClassifier
        #   Discretization is not instantiated with FunctionTransformer
        # {
        #    "Discretization": {"type": {"neq": "FunctionTransformer"}},
        #    "Classification": {"type": {"eq": "DecisionTreeClassifier"}},
        # },
        # if
        #   Feature Engineering is instantiated with SelectKBest with n_features eq 1 and
        #   Normalization is not instantiated with FunctionTransformer and
        #   Prototype is instantiated with FeaturesEngineering_Normalization_Classification
        {
            "Feature Engineering": {
                "type": {"eq": "SelectKBest"},
                "n_features": {"eq": 1},
            },
            "Normalization": {"type": {"neq": "FunctionTransformer"}},
            "Prototype": {"eq": "FeaturesEngineering_Normalization_Classification"},
        },
    ],
    "instance_constraints": [
        {
            "Feature Engineering": {
                "type": "SelectKBest",
                "n_features": 1,
            },
            "Normalization": {
                "type": "StandardScaler",
                "with_mean": True,
                "with_std": True,
            },
            "Prototype": "FeaturesEngineering_Normalization_Classification",
        },
        {
            "Feature Engineering": {
                "type": "SelectKBest",
                "n_features": 1,
            },
            "Normalization": {
                "type": "StandardScaler",
                "with_mean": True,
                "with_std": False,
            },
            "Prototype": "FeaturesEngineering_Normalization_Classification",
        },
        {
            "Feature Engineering": {
                "type": "SelectKBest",
                "n_features": 1,
            },
            "Normalization": {
                "type": "StandardScaler",
                "with_mean": False,
                "with_std": True,
            },
            "Prototype": "FeaturesEngineering_Normalization_Classification",
        },
        {
            "Feature Engineering": {
                "type": "SelectKBest",
                "n_features": 1,
            },
            "Normalization": {
                "type": "StandardScaler",
                "with_mean": False,
                "with_std": False,
            },
            "Prototype": "FeaturesEngineering_Normalization_Classification",
        },
        {
            "Feature Engineering": {
                "type": "SelectKBest",
                "n_features": 1,
            },
            "Normalization": {"type": "MinMaxScaler"},
            "Prototype": "FeaturesEngineering_Normalization_Classification",
        },
    ],
}


def check_step_assertion(step, operator_config, config):
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


def generate_template_constraint(constraint, config):
    for step, operator_config in constraint.items():
        if not check_step_assertion(step, operator_config, config):
            return False
    return True


def load_template_constraints(template_constraints):
    return [
        lambda config: generate_template_constraint(constraint, config)
        for constraint in template_constraints
    ]


def create_space(input_space):
    space = {}
    if type(input_space) is not dict:
        return input_space
    for key, value in input_space.items():
        if key == "choice":
            return tune.choice([create_space(elem) for elem in value])
        if type(value) is dict:
            space[key] = create_space(value)
        elif type(value) is list:
            raise Exception("You put an array without the 'choice' key")
        else:
            space[key] = value
    return space


print(create_space(input_space=prolog_output["space"]))
print(
    load_template_constraints(
        template_constraints=prolog_output["template_constraints"]
    )[0](
        {
            "Feature Engineering": {
                "type": "SelectKBest",
                "n_features": 1,
            },
            "Normalization": {"type": "a"},
            "Prototype": "FeaturesEngineering_Normalization_Classification",
        }
    )
)
