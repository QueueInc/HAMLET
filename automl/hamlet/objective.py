import copy
import time
import numpy as np

from collections import defaultdict

from fairlearn import metrics
from fairlearn.metrics._base_metrics import (
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.metrics._metric_frame import MetricFrame

from sklearn.model_selection import cross_validate, StratifiedKFold

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

# from sklearn.pipeline import Pipeline

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
from hamlet.transformers.lfr_wrapper import LFR_wrapper


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
        "mitigation" in prototype
        and "features" in prototype
        and prototype.index("mitigation") > prototype.index("features")
        and config["features"]["type"] == "PCA"
        and config["mitigation"]["type"] in ["CorrelationRemover", "LFR_wrapper"]
    ):
        raise Exception(f"""PCA before {config["mitigation"]["type"]}""")

    # if (
    #     config["discretization"]["type"] != "FunctionTransformer"
    #     and config["normalization"]["type"] != "FunctionTransformer"
    # ):
    #     raise Exception(
    #         "Discretization and Normalization are present in the same pipeline"
    #     )


def _get_indices_from_mask(mask, detect):
    return [i for i, x in enumerate(mask) if x == detect]


def _prepare_indexes(categorical_indicator, sensitive_indicator, feature_names):

    num_features = _get_indices_from_mask(categorical_indicator, False)
    cat_features = _get_indices_from_mask(categorical_indicator, True)

    return {
        "num_features": num_features,
        "cat_features": cat_features,
        "sen_num_features": [
            elem
            for elem in _get_indices_from_mask(sensitive_indicator, True)
            if elem in num_features
        ],
        "sen_cat_features": [
            elem
            for elem in _get_indices_from_mask(sensitive_indicator, True)
            if elem in cat_features
        ],
        "feature_names": feature_names,
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
        operator_parameters["sensitive_feature_ids"] = (
            indexes["sen_num_features"] + indexes["sen_cat_features"]
        )

    if config[step]["type"] == "LFR_wrapper":
        operator_parameters["prot_attr"] = [
            feature
            for idx, feature in enumerate(indexes["feature_names"])
            if idx in (indexes["sen_num_features"] + indexes["sen_cat_features"])
        ]
        operator_parameters["feature_names"] = indexes["feature_names"]

    return operator_parameters


def _prepare_operator(config, step, seed, indexes, operator_parameters):

    operator = globals()[config[step]["type"]](**operator_parameters)
    if "random_state" in operator.get_params():
        operator = globals()[config[step]["type"]](
            random_state=seed, **operator_parameters
        )

    if step not in ["discretization", "normalization", "encoding"]:
        return operator

    num_operator = (
        operator
        if step in ["discretization", "normalization"]
        else FunctionTransformer()
    )
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

    num_features = indexes["num_features"].copy()
    cat_features = indexes["cat_features"].copy()
    sen_num_features = indexes["sen_num_features"].copy()
    sen_cat_features = indexes["sen_cat_features"].copy()
    feature_names = indexes["feature_names"].copy()

    if step == "discretization":
        feature_names = [feature_names[feature] for feature in num_features] + [
            feature_names[feature] for feature in cat_features
        ]
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
        feature_names = [feature_names[feature] for feature in num_features] + [
            feature_names[feature] for feature in cat_features
        ]
        sen_num_features = [num_features.index(feature) for feature in sen_num_features]
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
            feature_names = [
                f"pca_{feature}"
                for feature in list(range(config[step]["n_components"]))
            ]
            num_features = list(range(config[step]["n_components"]))
            cat_features = []
            sen_num_features = []
            sen_cat_features = []
        elif config[step]["type"] == "SelectKBest":
            # selector = Pipeline(pipeline)
            # selector.fit_transform(X, y)
            selected_features = list(p_pipeline()[-1].get_support(indices=True))
            feature_names = [feature_names[feature] for feature in selected_features]
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
        feature_names = [
            feature
            for idx, feature in enumerate(feature_names)
            if idx not in (sen_num_features + sen_cat_features)
        ]
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
        "num_features": num_features,
        "cat_features": cat_features,
        "sen_num_features": sen_num_features,
        "sen_cat_features": sen_cat_features,
        "feature_names": feature_names,
    }


def _custom_metric(metric):

    from fairlearn.metrics import (
        selection_rate,
        true_positive_rate,
        false_positive_rate,
        MetricFrame,
    )

    def _equalized_odds(y_true, y_pred, sensitive_features, sample_weight=None):
        sel_rate = MetricFrame(
            selection_rate,
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_params={"sample_weight": sample_weight},
        )

        min_g = sel_rate.group_min()
        max_g = sel_rate.group_max()

        return 1 if max_g == 0 else min_g / max_g

    def _demographic_parity(y_true, y_pred, sensitive_features, sample_weight=None):
        fns = {"tpr": true_positive_rate, "fpr": false_positive_rate}
        sw_dict = {"sample_weight": sample_weight}
        sp = {"tpr": sw_dict, "fpr": sw_dict}
        sel_rate = MetricFrame(
            fns, y_true, y_pred, sensitive_features=sensitive_features, sample_params=sp
        )

        min_g = sel_rate.group_min()
        max_g = sel_rate.group_max()

        return min(result=[1 if b == 0 else a / b for a, b in zip(min_g, max_g)])

    def _equal_opportunity(y_true, y_pred, sensitive_features, sample_weight=None):
        sel_rate = MetricFrame(
            metrics=true_positive_rate,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            sample_params={"sample_weight": sample_weight},
        )

        min_g = sel_rate.group_min()
        max_g = sel_rate.group_max()

        return 1 if max_g == 0 else min_g / max_g

    return {
        "equalized_odds": _equalized_odds(),
        "demographic_parity": _demographic_parity(),
        "equal_opportunity": _equal_opportunity(),
    }[metric]


def _compute_fair_metric(
    fair_metric, X, y, sensitive_indicator, scores, skf, stratified_y
):

    # metrics_module = __import__("metrics")
    metrics_module = globals()["metrics"]
    metric_name = fair_metric
    performance_metric = getattr(metrics_module, metric_name)
    adjuster = lambda x: (1 - x) if "difference" in metric_name else x
    # performance_scorer = make_scorer(performance_metric)

    fair_scores = []
    fair_scores_by_group = []
    for fold, (train_indeces, test_indeces) in enumerate(skf.split(X, stratified_y)):
        # test_indeces = scores["indices"]["test"][fold]
        x_original = X.copy()[test_indeces, :]
        sensitive_mask = [i for i, x in enumerate(sensitive_indicator) if x == True]
        x_sensitive = x_original[:, sensitive_mask]

        x_sensitive = (
            x_sensitive if len(sensitive_mask) > 1 else x_sensitive.reshape(-1)
        )

        # fair_scores += [
        #     adjuster(
        #         performance_metric(
        #             y_true=np.array(y.copy()[test_indeces]),
        #             y_pred=np.array(scores["estimator"][fold].predict(x_original)),
        #             sensitive_features=x_sensitive,
        #         )
        #     )
        # ]

        y_true = np.array(y.copy()[test_indeces])
        y_pred = np.array(scores["estimator"][fold].predict(x_original))

        if metric_name.startswith("equalized_odds"):
            fns = {"tpr": true_positive_rate, "fpr": false_positive_rate}
        else:
            fns = selection_rate

        mf = MetricFrame(
            metrics=fns,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=x_sensitive,
        )

        method = "between_groups"  # or overall
        agg = "worst_case"  # or mean

        if metric_name.endswith("_ratio"):
            # ALTERNATIVA: utilizzare mf.overall
            ratios_by_group = mf.by_group / mf.by_group.max()
            if metric_name.startswith("equalized_odds"):
                if agg == "worst_case":
                    fair_score = min(mf.ratio(method=method))
                    fair_score_by_group = ratios_by_group.min(axis=1)
                else:
                    fair_score = mf.ratio(method=method).mean()
                    fair_score_by_group = ratios_by_group.mean(axis=1)
            else:
                fair_score = mf.ratio(method=method)
                fair_score_by_group = ratios_by_group
        else:
            # ALTERNATIVA: utilizzare mf.overall
            diffs_by_group = mf.by_group.max() - mf.by_group
            if metric_name.startswith("equalized_odds"):
                if agg == "worst_case":
                    fair_score = max(mf.difference(method=method))
                    fair_score_by_group = diffs_by_group.max(axis=1)
                else:
                    fair_score = mf.difference(method=method).mean()
                    fair_score_by_group = diffs_by_group.mean(axis=1)
            else:
                fair_score = mf.difference(method=method)
                fair_score_by_group = diffs_by_group

        fair_scores += [adjuster(fair_score)]

        # for lev in range(len(fair_score_by_group.index.levels)):
        #     fair_score_by_group.index.levels[lev] = fair_score_by_group.index.levels[
        #         lev
        #     ].astype(int)

        fair_score_by_group = fair_score_by_group.to_dict()
        fair_scores_by_group += [
            {
                key: adjuster(value)
                for key, value in fair_score_by_group.items()
            }
        ]

    merged = defaultdict(list)
    [merged[k].append(v) for d in fair_scores_by_group for k, v in d.items()]

    return fair_scores, dict(merged)


class Prototype:

    X = None
    y = None
    categorical_indicator = None
    sensitive_indicator = None
    encoding_mappings = None
    feature_names = None
    fair_metric = None
    metric = None
    mode = None
    buffer = None

    def __init__(
        self,
        X,
        y,
        categorical_indicator,
        sensitive_indicator,
        encoding_mappings,
        feature_names,
        fair_metric,
        metric,
        mode,
    ):
        self.X = X
        self.y = y
        self.categorical_indicator = categorical_indicator
        self.sensitive_indicator = sensitive_indicator
        self.encoding_mappings = [
            encoding_mappings[sens_feat] for sens_feat in sorted(encoding_mappings)
        ]
        self.feature_names = feature_names
        self.fair_metric = fair_metric
        self.metric = metric
        self.mode = mode
        self.buffer = Buffer()

    # We define the pipeline to evaluate
    def _instantiate_pipeline(self, seed, config):

        prototype = _get_prototype(config)
        _check_coherence(prototype, config)
        indexes = _prepare_indexes(
            self.categorical_indicator, self.sensitive_indicator, self.feature_names
        )

        pipeline = []
        for step in prototype:

            operator_parameters = _prepare_parameters(config, step, indexes)
            operator = _prepare_operator(
                config, step, seed, indexes, operator_parameters
            )
            pipeline.append([step, operator])

            def fit_pipeline():
                p = Pipeline(pipeline)
                p.fit_transform(self.X.copy(), self.y.copy())
                return p

            indexes = _adjust_indexes(step, config, indexes, lambda: fit_pipeline())

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
                self.buffer.printflush(f"The result for {config} was NaN")
                return True
            return False

        def _return_res(
            config, result, start_time, status, scores=None, add_to_buffer=True
        ):
            result["status"] = status
            _set_time(result, scores, start_time)
            if add_to_buffer:
                self.buffer.add_evaluation(config=config, result=result)
                self.buffer.printflush(f"{status}\n{config}\n{result}")
            return transform_result(result, self.metric, self.fair_metric, self.mode)

        config = transform_configuration(smac_config)

        result = {
            f"{self.fair_metric}": float("-inf"),
            f"{self.metric}": float("-inf"),
            "status": "fail",
            "total_time": 0,
            "fit_time": 0,
            "score_time": 0,
            "absolute_time": 0,
        }

        start_time = time.time()

        already_evaluated, reward = self.buffer.check_points_to_evaluate()
        if already_evaluated:
            return _return_res(
                config, reward, start_time, "already_evaluated", add_to_buffer=False
            )

        if self.buffer.check_template_constraints(config):
            return _return_res(config, result, start_time, "previous_constraint")

        try:

            scores = None
            pipeline = self._instantiate_pipeline(
                seed,
                config,
            )

            self.buffer.attach_timer(900)

            skf = StratifiedKFold(n_splits=5)
            sensistive_feature = _get_indices_from_mask(self.sensitive_indicator, True)
            sensitive_X = copy.deepcopy(self.X[:, sensistive_feature])
            if len(sensistive_feature) == 1:
                sensitive_X = sensitive_X.reshape(-1, 1)
            stratified_y = np.array(
                [
                    "".join([str(e) for e in elem])
                    for elem in np.concatenate(
                        [
                            sensitive_X,
                            self.y.reshape(-1, 1),
                        ],
                        axis=1,
                    )
                ]
            )
            scores = cross_validate(
                pipeline,
                self.X.copy(),
                self.y.copy(),
                scoring=[self.metric],
                cv=skf.split(self.X, stratified_y),
                return_estimator=True,
                return_train_score=False,
                # return_indices=True,
                verbose=0,
            )

            self.buffer.detach_timer()
            fair_scores, fair_scores_by_group = _compute_fair_metric(
                self.fair_metric,
                self.X.copy(),
                self.y.copy(),
                self.sensitive_indicator,
                scores,
                skf,
                stratified_y,
            )

            res = {
                f"{self.metric}": scores["test_" + self.metric],
                f"{self.fair_metric}": fair_scores,
            }

            for current_metric in [self.fair_metric, self.metric]:
                result[f"flatten_{current_metric}"] = "_".join(
                    [str(round(score, 2)) for score in res[current_metric]]
                )
            result["by_group"] = {
                "_".join(
                    [
                        self.encoding_mappings[sens_feat][int(sens_group)]
                        for sens_feat, sens_group in enumerate(key)
                    ]
                ): round(np.mean(value), 2)
                for key, value in fair_scores_by_group.items()
            }

            if any([_res(m, r) for m, r in res.items()]):
                raise Exception(f"The result for {config} was NaN")

            result["status"] = "success"

        except TimeException:
            self.buffer.printflush("Timeout")
        except Exception as e:
            self.buffer.detach_timer()
            self.buffer.printflush(f"Something went wrong:\n{e}")

        return _return_res(config, result, start_time, result["status"], scores=scores)
