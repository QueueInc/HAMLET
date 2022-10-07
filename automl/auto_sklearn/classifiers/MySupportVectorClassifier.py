from typing import Optional

import resource
import sys

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import softmax
from autosklearn.util.common import check_for_bool, check_none


class MySupportVectorClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(
        self,
        C,
        kernel,
        degree,
        gamma,
        shrinking,
        random_state=None,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.shrinking = shrinking
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.svm

        self.C = float(self.C)
        self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)

        self.shrinking = check_for_bool(self.shrinking)

        self.estimator = sklearn.svm.SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            shrinking=self.shrinking,
            random_state=self.random_state,
        )
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MySupportVectorClassifier",
            "name": "My Support Vector Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        C = CategoricalHyperparameter(
            "C", choices=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
        )
        kernel = CategoricalHyperparameter(
            name="kernel", choices=["linear", "rbf", "poly", "sigmoid"]
        )
        degree = CategoricalHyperparameter("degree", choices=[2, 3, 4, 5, 10, 20])
        gamma = CategoricalHyperparameter("gamma", choices=["auto", "scale"])
        shrinking = CategoricalHyperparameter("shrinking", ["True", "False"])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, shrinking])

        return cs
