# Scikit-learn provides a set of machine learning techniques
import json
import os
import sys
import time
import gc

import numpy as np

from fairlearn import metrics

# from fairlearn.metrics import demographic_parity_ratio

# from sklearn import metrics
# from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer

## Feature Engineering operators
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer

## Normalization operators
from sklearn.preprocessing import (
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    KBinsDiscretizer,
    Binarizer,
)

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

## Classification algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .buffer import Buffer, TimeException


def get_prototype(config):
    # We define the ml pipeline to optimize (i.e., the order of the pre-processing transformations + the ml algorithm)
    ml_pipeline = config["prototype"]
    if ml_pipeline is None:
        raise NameError("No prototype specified")
    else:
        ml_pipeline = ml_pipeline.split("_")
    return ml_pipeline


def get_indices_from_mask(mask, detect):
    return [i for i, x in enumerate(mask) if x == detect]


def instantiate_pipeline(prototype, categorical_indicator, X, y, seed, config):
    num_features = get_indices_from_mask(categorical_indicator, False)
    cat_features = get_indices_from_mask(categorical_indicator, True)
    # if (
    #     config["discretization"]["type"] != "FunctionTransformer"
    #     and config["normalization"]["type"] != "FunctionTransformer"
    # ):
    #     raise Exception(
    #         "Discretization and Normalization are present in the same pipeline"
    #     )

    # In such a precise order:
    pipeline = []
    for step in prototype:
        operator_parameters = {
            param_name: config[step][param_name]
            for param_name in config[step]
            if param_name != "type"
        }
        if config[step]["type"] == "MLPClassifier":
            operator_parameters["hidden_layer_sizes"] = (
                operator_parameters["n_neurons"]
            ) * operator_parameters["n_hidden_layers"]
            operator_parameters.pop("n_neurons", None)
            operator_parameters.pop("n_hidden_layers", None)

        # we instantiate the operator/algorithm,
        if "random_state" in globals()[config[step]["type"]]().get_params():
            operator = globals()[config[step]["type"]](
                random_state=seed, **operator_parameters
            )
        else:
            operator = globals()[config[step]["type"]](**operator_parameters)
        # and we add it to the pipeline
        if step in ["discretization", "normalization", "encoding"]:
            if step in ["discretization", "normalization"]:
                num_operator = operator
                cat_operator = FunctionTransformer()
            else:
                num_operator = FunctionTransformer()
                cat_operator = operator

            pipeline.append(
                [
                    step,
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline(steps=[(f"{step}_num", num_operator)]),
                                num_features.copy(),
                            ),
                            (
                                "cat",
                                Pipeline(steps=[(f"{step}_cat", cat_operator)]),
                                cat_features.copy(),
                            ),
                        ]
                    ),
                ]
            )
        else:
            pipeline.append([step, operator])

        if step == "discretization":
            cat_features = list(range(len(cat_features + num_features)))
            num_features = []
        elif step in ["encoding", "normalization"]:
            num_features = list(range(len(num_features)))
            cat_features = list(
                range(len(num_features), len(num_features) + len(cat_features))
            )
        elif step == "features":
            if config[step]["type"] == "PCA":
                num_features = list(range(config[step]["n_components"]))
                cat_features = []
            elif config[step]["type"] == "SelectKBest":
                selector = Pipeline(pipeline)
                selector.fit_transform(X, y)
                selected_features = list(selector[-1].get_support(indices=True))
                num_features = [
                    selected_features.index(feature)
                    for feature in num_features
                    if feature in selected_features
                ]
                cat_features = [
                    selected_features.index(feature)
                    for feature in cat_features
                    if feature in selected_features
                ]
    return Pipeline(pipeline)


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


# We define the function to optimize
def objective(
    X,
    y,
    categorical_indicator,
    sensitive_indicator,
    fair_metric,
    metric,
    seed,
    smac_config,
):
    def set_time(result, scores, start_time):
        result["absolute_time"] = time.time()
        result["total_time"] = result["absolute_time"] - start_time

        if scores and "fit_time" in scores:
            result["fit_time"] = np.mean(scores["fit_time"])
        if scores and "score_time" in scores:
            result["score_time"] = np.mean(scores["score_time"])

    config = transform_configuration(smac_config)

    result = {
        fair_metric: float("-inf"),
        metric: float("-inf"),
        "status": "fail",
        "total_time": 0,
        "fit_time": 0,
        "score_time": 0,
        "absolute_time": 0,
    }

    start_time = time.time()
    scores = None

    # We check if the point has already been evaluated
    # (i.e., if it is in the "points_to_evaluate" read by the json)
    is_point_to_evaluate, reward = Buffer().check_points_to_evaluate()
    if is_point_to_evaluate:
        return reward

    if Buffer().check_template_constraints(config):
        result["status"] = "previous_constraint"
        set_time(result, scores, start_time)
        Buffer().add_evaluation(config=config, result=result)
        return result

    Buffer().printflush(config)

    try:

        X_copy = X.copy()
        y_copy = y.copy()
        X_copy_ii = X.copy()
        y_copy_ii = y.copy()

        prototype = get_prototype(config)
        pipeline = instantiate_pipeline(
            prototype,
            categorical_indicator,
            X_copy,
            y_copy,
            seed,
            config,
        )

        Buffer().printflush("opt")
        Buffer().attach_timer(900)

        cv = 5
        scores = cross_validate(
            pipeline,
            X_copy_ii,
            y_copy_ii,
            scoring=[metric],
            # scoring={
            #     metric: performance_scorer,
            #     "demographic_parity": make_scorer(
            #         demographic_parity_ratio, sensitive_features=[]
            #     ),
            # },
            cv=cv,
            return_estimator=True,
            return_train_score=False,
            return_indices=True,
            verbose=0,
        )

        Buffer().detach_timer()
        Buffer().printflush("end opt")

        # Buffer().printflush(scores["estimator"])
        # Buffer().printflush(type(scores["estimator"]))
        # log = [estimator.__dict__ for estimator in scores["estimator"]]
        # with open(
        #     os.path.join(
        #         "/", "home", "results", "trial", "automl", "output", "log.json"
        #     ),
        #     "w",
        # ) as outfile:
        #     json.dump(log, outfile)

        # metrics_module = __import__("metrics")
        metrics_module = globals()["metrics"]
        performance_metric = getattr(metrics_module, f"{fair_metric}_ratio")
        # performance_scorer = make_scorer(performance_metric)

        fair_scores = []
        for fold in range(cv):
            test_indeces = scores["indices"]["test"][fold]
            x_original = X.copy()[test_indeces, :]
            x_sensitive = x_original[
                :, get_indices_from_mask(sensitive_indicator, True)
            ]
            # Buffer().printflush(x_original)
            # Buffer().printflush(x_sensitive)
            # Buffer().printflush(y.copy()[test_indeces])
            # Buffer().printflush(scores["estimator"][fold].predict(x_original))
            # Buffer().printflush(
            #     performance_metric(
            #         y_true=np.array(y.copy()[test_indeces]).reshape(-1, 1),
            #         y_pred=np.array(
            #             scores["estimator"][fold].predict(x_original)
            #         ).reshape(-1, 1),
            #         sensitive_features=x_sensitive,
            #     )
            # )

            # forse fare .reshape(-1, 1) in caso di intersectionality
            fair_scores += [
                performance_metric(
                    y_true=np.array(y.copy()[test_indeces]),
                    y_pred=np.array(scores["estimator"][fold].predict(x_original)),
                    sensitive_features=[str(elem) for elem in x_sensitive.reshape(-1)],
                )
            ]

        result[metric] = np.mean(scores["test_" + metric])
        result[f"flatten_{fair_metric}"] = "_".join(
            [str(score) for score in fair_scores]
        )
        result[f"{fair_metric}"] = np.mean(fair_scores)
        if np.isnan(result[f"{fair_metric}"]):
            result[f"{fair_metric}"] = float("-inf")
        if np.isnan(result[metric]):
            result[metric] = float("-inf")
            print(f"The result for {config} was NaN")
            raise Exception(f"The result for {config} was NaN")
        result["status"] = "success"

    except TimeException:
        Buffer().printflush("Timeout")
    except Exception as e:
        Buffer().detach_timer()
        Buffer().printflush("#########################################")
        Buffer().printflush("Something went wrong")
        Buffer().printflush(str(e))
    finally:
        set_time(result, scores, start_time)

        del X_copy
        del X_copy_ii
        del y_copy
        del y_copy_ii

        gc.collect()

    Buffer().add_evaluation(config=config, result=result)

    # TODO
    # Transform result
    # Per esempio: smac gestisce solo min! Fare 1 - metric
    # Controlla il formato (il dict è composto uguale? Non vuole solo le metriche interessate?)

    return result
