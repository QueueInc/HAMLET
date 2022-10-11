from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA


class MyAdaboostClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, n_estimators, learning_rate, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.ensemble
        import sklearn.tree

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)

        estimator = sklearn.ensemble.AdaBoostClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )

        estimator.fit(X, Y)

        self.estimator = estimator
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
            "shortname": "MyAdaBoostClassifier",
            "name": "My AdaBoost Classifier",
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
        cs = ConfigurationSpace()

        n_estimators = CategoricalHyperparameter(
            name="n_estimators", choices=[10, 50, 100, 500]
        )

        learning_rate = CategoricalHyperparameter(
            name="learning_rate", choices=[0.0001, 0.001, 0.01, 0.1, 1.0]
        )

        cs.add_hyperparameters([n_estimators, learning_rate])
        return cs
