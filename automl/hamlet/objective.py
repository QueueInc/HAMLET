# Scikit-learn provides a set of machine learning techniques
import traceback
import time
import numpy as np
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .buffer import Buffer


def get_prototype(config):
    # We define the ml pipeline to optimize (i.e., the order of the pre-processing transformations + the ml algorithm)
    ml_pipeline = config["prototype"]
    if ml_pipeline is None:
        raise NameError("No prototype specified")
    else:
        ml_pipeline = ml_pipeline.split("_")
    return ml_pipeline


def instantiate_pipeline(prototype, categorical_indicator, X, y, seed, config):
    num_features = [i for i, x in enumerate(categorical_indicator) if x == False]
    cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
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


# We define the function to optimize
def objective(X, y, categorical_indicator, metric, seed, config):
    result = {metric: float("-inf"), "status": "fail", "time": 0}

    is_point_to_evaluate, reward = Buffer().check_points_to_evaluate(config)
    if is_point_to_evaluate:
        return reward

    if Buffer().check_template_constraints(config):
        result["status"] = "previous_constraint"
        Buffer().add_evaluation(config=config, result=result)
        return result

    try:
        prototype = get_prototype(config)

        pipeline = instantiate_pipeline(
            prototype, categorical_indicator, X, y, seed, config
        )

        scores = cross_validate(
            pipeline,
            X.copy(),
            y.copy(),
            scoring=[metric],
            cv=10,
            return_estimator=False,
            return_train_score=False,
            verbose=0,
        )

        result[metric] = np.mean(scores["test_" + metric])
        if np.isnan(result[metric]):
            result[metric] = float("-inf")
            raise Exception(f"The result for {config} was NaN")
        result["status"] = "success"
        result["time"] = time.time()

    except Exception as e:
        print(
            f"""
            BEGIN EXCEPTION
            {"_"*100}
            {e}
            {"_"*100}
            {traceback.print_exc()}
            {"_"*100}
            END EXCEPTION
            """
        )

    Buffer().add_evaluation(config=config, result=result)
    return result
