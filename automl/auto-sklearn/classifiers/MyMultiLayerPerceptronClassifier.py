from typing import Optional

import copy

import numpy as np
from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponent,
)
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.util.common import check_for_bool


class MyMultiLayerPerceptronClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(
        self,
        hidden_layer_depth,
        num_nodes_per_layer,
        activation,
        solver,
        alpha,
        learning_rate,
        validation_fraction=None,
        random_state=None,
        verbose=0,
    ):
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_iter_

    def fit(self, X, y):

        from sklearn.neural_network import MLPClassifier

        self.hidden_layer_depth = int(self.hidden_layer_depth)
        self.num_nodes_per_layer = int(self.num_nodes_per_layer)
        self.hidden_layer_sizes = tuple(
            self.num_nodes_per_layer for i in range(self.hidden_layer_depth)
        )
        self.activation = str(self.activation)
        self.alpha = float(self.alpha)
        self.solver = self.solver

        self.estimator = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            random_state=copy.copy(self.random_state),
            verbose=self.verbose,
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
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MyMultiLayerPerceptronClassifier",
            "name": "My Multilayer Percepton Classifier",
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
        hidden_layer_depth = CategoricalHyperparameter(
            name="hidden_layer_depth",
            choices=[1, 5, 10, 25],
        )
        num_nodes_per_layer = CategoricalHyperparameter(
            name="num_nodes_per_layer",
            choices=[5, 10, 25, 50, 100],
        )
        activation = CategoricalHyperparameter(
            name="activation", choices=["logistic", "tanh", "relu"]
        )
        solver = CategoricalHyperparameter(
            name="solver", choices=["lbfgs", "sgd", "adam"]
        )
        alpha = CategoricalHyperparameter(
            name="alpha", choices=[0.0001, 0.001, 0.01, 0.00001]
        )
        learning_rate = CategoricalHyperparameter(
            name="learning_rate", choices=["constant", "invscaling", "adaptive"]
        )

        cs.add_hyperparameters(
            [
                hidden_layer_depth,
                num_nodes_per_layer,
                activation,
                solver,
                alpha,
                learning_rate,
            ]
        )
        return cs
