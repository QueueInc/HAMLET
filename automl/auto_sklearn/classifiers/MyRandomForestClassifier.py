from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from autosklearn.util.common import check_for_bool, check_none


class MyRandomForestClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(
        self,
        n_estimators,
        max_depth,
        max_features,
        min_samples_split,
        max_leaf_nodes,
        bootstrap,
        criterion,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.estimator = None
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier

        self.n_estimators = int(self.n_estimators)
        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)

        self.min_samples_split = int(self.min_samples_split)
        self.max_features = int(self.max_features)
        self.bootstrap = check_for_bool(self.bootstrap)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            max_leaf_nodes=self.max_leaf_nodes,
            bootstrap=self.bootstrap,
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
            "shortname": "MyRandomForestClassifier",
            "name": "My Random Forest Classifier",
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

        n_estimators = CategoricalHyperparameter("n_estimators", [10, 25, 50, 75, 100])
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 5)
        max_features = UniformIntegerHyperparameter("max_features", 1, 4)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 6)
        max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 2, 6)
        bootstrap = CategoricalHyperparameter("bootstrap", ["True", "False"])
        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"])

        cs.add_hyperparameters(
            [
                n_estimators,
                max_depth,
                max_features,
                min_samples_split,
                max_leaf_nodes,
                bootstrap,
                criterion,
            ]
        )
        return cs
