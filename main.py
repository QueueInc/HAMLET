# System utils to set the seed
import sys

import traceback

# OpenML provides several benchmark datasets
import openml

# Numpy provides useful data structure
import numpy as np
from numpy import dtype

# Matplotlib allows to plot charts
import matplotlib.pyplot as plt

# Seaborn provides different data analysis tools
import seaborn as sns

# Pandas provides a tabular view of the data
import pandas as pd

# Scipy provides math utilities
from scipy.stats import pearsonr

# Hyperopt and FLAML provide Bayesian Optimization techniques
from flaml import tune

# Scikit-learn provides a set of machine learning techniques
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer

## Feature Engineering operators
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

## Normalization operators
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
)

## Classification algorithms
from sklearn.neighbors import KNeighborsClassifier


def get_dataset(name):
    loader = {
        "blood": 1464,
        "breast": 1465,  # this is breast-tissue, not breast cancer
        "diabetes": 37,
        "ecoli": 40671,
        "iris": 61,
        "parkinsons": 1488,
        "seeds": 1499,
        "thyroid": 40682,
        "vehicle": 54,
        "wine": 187,
    }
    if name in loader:
        return load_dataset_from_openml(loader[name])
    else:
        raise Exception("There is no such a dataset in the loader")


def load_dataset_from_openml(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    dataset_features_names = [str(elem) for elem in list(dataset.features.values())]
    dataset_features_names = dataset_features_names[1:]
    return X, y, dataset_features_names


# We load the wine dataset
X, y, dataset_features_names = get_dataset("wine")

# We get the pandas dataframe
X_df = pd.DataFrame(X, columns=dataset_features_names)
y_df = pd.DataFrame(y, columns=["target"])
dataset_df = pd.concat([X_df, y_df], axis=1)

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
ml_pipeline_space = {
    "MLPipeline": tune.choice(
        [
            "Normalization_FeatureEngineering_Classification",
            "FeatureEngineering_Normalization_Classification",
        ]
    )
}

space = {
    **features_enginerring_space,
    **normalization_space,
    **classification_space,
    **ml_pipeline_space,
}

buffer_confs = {}
buffer_results = {}

i = 0


# We define the function to optimize
def objective(conf):
    global i
    result = {"accuracy": float("-inf"), "status": "fail"}
    # print(f'''conf:
    # {conf}''')
    try:
        # We define the ml pipeline to optimize (i.e., the order of the pre-processing transformations + the ml algorithm)
        ml_pipeline = conf["MLPipeline"]
        if ml_pipeline is None:
            raise NameError("No ML pipeline specified")
        else:
            ml_pipeline = ml_pipeline.split("_")

        # In such a precise order:
        pipeline = []
        for step in ml_pipeline:
            # we define the parametrization of each step,
            # print(f'''conf[{step}]:
            # {conf[step]}''')
            operator_parameters = {
                param_name: conf[step][param_name]
                for param_name in conf[step]
                if param_name != "type"
            }
            # we instantiate the operator/algorithm,
            if "random_state" in globals()[conf[step]["type"]]().get_params():
                operator = globals()[conf[step]["type"]](
                    random_state=seed, **operator_parameters
                )
            else:
                operator = globals()[conf[step]["type"]](**operator_parameters)
            # and we add it to the pipeline
            pipeline.append([step, operator])
        pipeline = Pipeline(pipeline)
        # We evaluate the pipeline with k-cross validarion
        scores = cross_validate(
            pipeline,
            X.copy(),
            y.copy(),
            scoring=["accuracy"],
            cv=10,
            return_estimator=False,
            return_train_score=False,
            verbose=0,
        )
        # We get the accuracy
        accuracy = np.mean(scores["test_accuracy"])
        result["accuracy"] = accuracy
        result["status"] = "success"
    except Exception as e:
        print(
            f"""MyException: {e}
              {traceback.print_exc()}"""
        )
    buffer_confs[i] = conf
    buffer_results[i] = result
    i += 1
    return result


batch_size = 10

points_to_evaluate = list(buffer_confs.values())
evaluated_rewards = [result["accuracy"] for result in buffer_results.values()]

# We set the seed for reproducible results
seed = 42
np.random.seed(seed)
# We run SMBO for each of the optimizations
analysis = tune.run(
    evaluation_function=objective,
    config=space,
    metric="accuracy",
    mode="max",
    num_samples=batch_size,
    points_to_evaluate=points_to_evaluate,
    evaluated_rewards=evaluated_rewards,
    verbose=False,
)

print(analysis.best_result)
