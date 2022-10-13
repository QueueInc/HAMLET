from typing import Optional

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from autosklearn.util.common import check_none


class MyDecisionTreeClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(
        self,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        max_leaf_nodes,
        splitter,
        criterion,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.splitter = splitter
        self.criterion = criterion
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeClassifier

        self.max_features = float(self.max_features)
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.estimator = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            splitter=self.splitter,
            criterion=self.criterion,
            random_state=self.random_state,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MyDecisionTreeClassifier",
            "name": "My Decision Tree Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        max_depth = UniformIntegerHyperparameter("max_depth", 1, 5)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 6)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf",
            1,
            6,
        )
        max_features = UniformIntegerHyperparameter(
            "max_features",
            1,
            4,
        )
        max_leaf_nodes = UniformIntegerHyperparameter(
            "max_leaf_nodes",
            2,
            6,
        )
        splitter = CategoricalHyperparameter("splitter", ["best", "random"])
        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"])

        cs.add_hyperparameters(
            [
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features,
                max_leaf_nodes,
                splitter,
                criterion,
            ]
        )

        return cs
