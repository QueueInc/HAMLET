import time
import numpy as np

from fairlearn import metrics
from sklearn.model_selection import cross_validate

## Base operators
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

## Feature Engineering operators
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

## Imputation operators
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

# Rebalancing operators
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

## Mitigation operators
from fairlearn.preprocessing import CorrelationRemover

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


from hamlet.buffer import Buffer, TimeException
from hamlet.utils.flaml_to_smac import transform_configuration, transform_result


def _get_prototype(config):
    # We define the ml pipeline to optimize (i.e., the order of the pre-processing transformations + the ml algorithm)
    ml_pipeline = config["prototype"]
    if ml_pipeline is None:
        raise NameError("No prototype specified")
    else:
        ml_pipeline = ml_pipeline.split("_")
    return ml_pipeline


def _check_coherence(prototype, config):
    
    if (
        prototype.index("mitigation") > prototype.index("features")
        and config["features"]["type"] == "PCA"
        and config["mitigation"]["type"] == "CorrelationRemover"
    ):
        raise Exception("PCA before CorrelationRemover")

    # if (
    #     config["discretization"]["type"] != "FunctionTransformer"
    #     and config["normalization"]["type"] != "FunctionTransformer"
    # ):
    #     raise Exception(
    #         "Discretization and Normalization are present in the same pipeline"
    #     )


def _prepare_indexes(categorical_indicator, sensitive_indicator):

    def _get_indices_from_mask(mask, detect):
        return [i for i, x in enumerate(mask) if x == detect]

    num_features = _get_indices_from_mask(categorical_indicator, False)
    cat_features = _get_indices_from_mask(categorical_indicator, True)

    return {
        "num_features" : num_features,
        "cat_features" : cat_features,
        "sen_num_features" : [
            elem
            for elem in _get_indices_from_mask(sensitive_indicator, True)
            if elem in num_features
        ],
        "sen_cat_features" : [
            elem
            for elem in _get_indices_from_mask(sensitive_indicator, True)
            if elem in cat_features
        ]
    }


def _prepare_parameters(config, step, indexes):
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

    if config[step]["type"] == "CorrelationRemover":
        operator_parameters["sensitive_feature_ids"] = indexes["sen_num_features"] + indexes["sen_cat_features"]
    
    return operator_parameters


def _prepare_operator(config, step, seed, indexes, operator_parameters):

    operator = globals()[config[step]["type"]](**operator_parameters)
    if "random_state" in operator.get_params():
        operator = globals()[config[step]["type"]](
            random_state=seed, **operator_parameters
        )

    if step not in ["discretization", "normalization", "encoding"]:
        return operator

    num_operator = operator if step in ["discretization", "normalization"] else FunctionTransformer()
    cat_operator = operator if step in ["encoding"] else FunctionTransformer()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[(f"{step}_num", num_operator)]),
                indexes["num_features"].copy(),
            ),
            (
                "cat",
                Pipeline(steps=[(f"{step}_cat", cat_operator)]),
                indexes["cat_features"].copy(),
            ),
        ]
    )


def _adjust_indexes(step, config, indexes, p_pipeline):

    num_features = indexes["num_features"]
    cat_features = indexes["cat_features"]
    sen_num_features = indexes["sen_num_features"]
    sen_cat_features = indexes["sen_cat_features"]

    if step == "discretization":
        sen_cat_features = [
            num_features.index(feature) for feature in sen_num_features
        ] + [
            cat_features.index(feature) + len(num_features)
            for feature in sen_cat_features
        ]
        sen_num_features = []
        cat_features = list(range(len(cat_features + num_features)))
        num_features = []
    elif step in ["encoding", "normalization"]:
        sen_num_features = [
            num_features.index(feature) for feature in sen_num_features
        ]
        sen_cat_features = [
            cat_features.index(feature) + len(num_features)
            for feature in sen_cat_features
        ]
        num_features = list(range(len(num_features)))
        cat_features = list(
            range(len(num_features), len(num_features) + len(cat_features))
        )
    elif step == "features":
        if config[step]["type"] == "PCA":
            num_features = list(range(config[step]["n_components"]))
            cat_features = []
            sen_num_features = []
            sen_cat_features = []
        elif config[step]["type"] == "SelectKBest":
            # selector = Pipeline(pipeline)
            # selector.fit_transform(X, y)
            selected_features = list(p_pipeline()[-1].get_support(indices=True))
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
            sen_num_features = [
                selected_features.index(feature)
                for feature in sen_num_features
                if feature in selected_features
            ]
            sen_cat_features = [
                selected_features.index(feature)
                for feature in sen_cat_features
                if feature in selected_features
            ]
    elif step == "mitigation":
        num_features = list(
            range(
                len(cat_features + num_features)
                - len(sen_num_features + sen_cat_features)
            )
        )
        cat_features = []
        sen_num_features = []
        sen_cat_features = []

    return {
        "num_features" : num_features,
        "cat_features" : cat_features,
        "sen_num_features" : sen_num_features,
        "sen_cat_features" : sen_cat_features
    }


def _compute_fair_metric(fair_metric, X, y, sensitive_indicator, scores):

    # metrics_module = __import__("metrics")
    metrics_module = globals()["metrics"]
    performance_metric = getattr(metrics_module, f"{fair_metric}_ratio")
    # performance_scorer = make_scorer(performance_metric)

    fair_scores = []
    for fold in range(len(scores["indices"]["test"])):
        test_indeces = scores["indices"]["test"][fold]
        x_original = X.copy()[test_indeces, :]
        x_sensitive = x_original[
            :, [i for i, x in enumerate(sensitive_indicator) if x == True]
        ]

        # forse fare .reshape(-1, 1) in caso di intersectionality
        fair_scores += [
            performance_metric(
                y_true=np.array(y.copy()[test_indeces]),
                y_pred=np.array(scores["estimator"][fold].predict(x_original)),
                sensitive_features=[
                    str(elem) for elem in x_sensitive.reshape(-1)
                ],
            )
        ]
    
    return fair_scores


class Prototype:

    X = None
    y = None
    categorical_indicator = None
    sensitive_indicator = None
    fair_metric = None
    metric = None
    mode = None

    def __init__(
        self,
        X,
        y,
        categorical_indicator,
        sensitive_indicator,
        fair_metric,
        metric,
        mode,
    ):
        self.X = X
        self.y = y
        self.categorical_indicator = categorical_indicator
        self.sensitive_indicator = sensitive_indicator
        self.fair_metric = fair_metric
        self.metric = metric
        self.mode = mode

    
    # We define the pipeline to evaluate
    def _instantiate_pipeline(
        self, seed, config
    ):
        
        prototype = _get_prototype(config)
        _check_coherence(prototype, config)
        indexes = _prepare_indexes(self.categorical_indicator, self.sensitive_indicator)

        pipeline = []
        for step in prototype:
            
            operator_parameters = _prepare_parameters(config, step, indexes)
            operator = _prepare_operator(config, step, seed, indexes, operator_parameters)
            pipeline.append([step, operator])

            def fit_pipeline():
                p = Pipeline(pipeline)
                p.fit_transform(self.X.copy(), self.y.copy())
                return p

            indexes = _adjust_indexes(step, config, indexes, lambda : fit_pipeline())

        return Pipeline(pipeline)


    # We define the function to optimize
    def objective(
        self,
        smac_config,
        seed,
    ):
        def _set_time(result, scores, start_time):
            result["absolute_time"] = time.time()
            result["total_time"] = result["absolute_time"] - start_time

            if scores and "fit_time" in scores:
                result["fit_time"] = np.mean(scores["fit_time"])
            if scores and "score_time" in scores:
                result["score_time"] = np.mean(scores["score_time"])

        def _res(m, r):
            result[m] = np.mean(r)
            if np.isnan(result[m]):
                result[m] = float("-inf")
                Buffer().printflush(f"The result for {config} was NaN")
                return True
            return False
        
        def _return_res(config, result, start_time, status, scores=None, add_to_buffer=True):
            result["status"] = status
            _set_time(result, scores, start_time)
            if add_to_buffer:
                Buffer().add_evaluation(config=config, result=result)
            Buffer().printflush(f"{status}\n{config}\n{result}")
            return transform_result(
                result, self.metric, self.fair_metric, self.mode
            )
            
        config = transform_configuration(smac_config)

        result = {
            f"{self.fair_metric}" : float("-inf"),
            f"{self.metric}" : float("-inf"),
            "status": "fail",
            "total_time": 0,
            "fit_time": 0,
            "score_time": 0,
            "absolute_time": 0,
        }

        start_time = time.time()

        already_evaluated, reward = Buffer().check_points_to_evaluate()
        if already_evaluated:
            return _return_res(config, reward, start_time, "already_evaluated", add_to_buffer=False)
        

        if Buffer().check_template_constraints(config):
            return _return_res(config, result, start_time, "previous_constraint")


        try:

            scores = None
            pipeline = self._instantiate_pipeline(
                seed,
                config,
            )

            Buffer().attach_timer(900)

            # TODO
            # Potrebbe valere la pena fare la stratificazione sulla feature sensibile
            scores = cross_validate(
                pipeline,
                self.X.copy(),
                self.y.copy(),
                scoring=[self.metric],
                cv=5,
                return_estimator=True,
                return_train_score=False,
                return_indices=True,
                verbose=0,
            )

            Buffer().detach_timer()

            res = {
                f"{self.metric}" : scores["test_" + self.metric],
                f"{self.fair_metric}" : _compute_fair_metric(self.fair_metric, self.X.copy(), self.y.copy(), self.sensitive_indicator, scores)
            }

            result[f"flatten_{self.fair_metric}"] = "_".join(
                [str(score) for score in res[self.fair_metric]]
            )

            if any([_res(m, r) for m, r in res.items()]):
                raise Exception(f"The result for {config} was NaN")

            result["status"] = "success"

        except TimeException:
            Buffer().printflush("Timeout")
        except Exception as e:
            Buffer().detach_timer()
            Buffer().printflush(f"Something went wrong:\n{e}")
        
        return _return_res(config, result, start_time, result["status"], scores=scores)
