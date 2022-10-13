from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA


class MyKNearestNeighborsClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, n_neighbors, weights, metric, random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, Y):
        import sklearn.multiclass
        import sklearn.neighbors

        estimator = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric
        )

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MyKNearestNeighborsClassifier",
            "name": "My K-Nearest Neighbor Classification",
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

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=3, upper=20
        )
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"]
        )
        metric = CategoricalHyperparameter(
            name="metric", choices=["minkowski", "euclidean", "manhattan"]
        )
        cs.add_hyperparameters([n_neighbors, weights, metric])

        return cs
