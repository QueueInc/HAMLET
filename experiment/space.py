# FLAML provides Bayesian Optimization techniques
from flaml import tune


def get_space():
    ## Normalization
    normalization_space = {
        "Normalization": tune.choice(
            [
                {
                    "type": "FunctionTransformer",
                },
                {
                    "type": "StandardScaler",
                    "with_mean": tune.choice([True, False]),
                    "with_std": tune.choice([True, False]),
                },
                # {
                #    'type': 'PowerTransformer',
                # },
                {
                    "type": "MinMaxScaler",
                },
                {
                    "type": "RobustScaler",
                    #'quantile_range': tune.choice([(25.0, 75.0),(10.0, 90.0), (5.0, 95.0)]),
                    "with_centering": tune.choice([True, False]),
                    "with_scaling": tune.choice([True, False]),
                },
            ]
        ),
    }

    ## Feature Engineering
    features_enginerring_space = {
        "FeatureEngineering": tune.choice(
            [
                {
                    "type": "FunctionTransformer",
                },
                {
                    "type": "PCA",
                    "n_components": tune.choice([1, 2, 3, 4]),
                },
                {
                    "type": "SelectKBest",
                    "k": tune.choice([1, 2, 3, 4]),
                },
            ]
        ),
    }

    ## Classification
    classification_space = {
        "Classification": tune.choice(
            [
                {
                    "type": "KNeighborsClassifier",
                    "n_neighbors": tune.choice([3, 5, 7, 9, 11, 13, 15, 17, 19]),
                    "weights": tune.choice(["uniform", "distance"]),
                    "metric": tune.choice(["minkowski", "euclidean", "manhattan"]),
                }
            ]
        ),
    }

    ## Order of the transformations
    prototype_space = {
        "Prototype": tune.choice(
            [
                "Normalization_FeatureEngineering_Classification",
                "FeatureEngineering_Normalization_Classification",
            ]
        )
    }

    return {
        **features_enginerring_space,
        **normalization_space,
        **classification_space,
        **prototype_space,
    }
