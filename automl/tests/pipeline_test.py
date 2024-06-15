import unittest

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler,
    PowerTransformer, KBinsDiscretizer, Binarizer
)
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from fairlearn.preprocessing import CorrelationRemover
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from unittest.mock import MagicMock

from automl.hamlet.objective import _get_prototype, _check_coherence, _prepare_indexes, _prepare_parameters, _prepare_operator, _adjust_indexes

class TestPipelineFunctions(unittest.TestCase):

    def test_get_prototype_valid(self):
        config = {"prototype": "step1_step2_step3"}
        result = _get_prototype(config)
        self.assertEqual(result, ["step1", "step2", "step3"])

    def test_get_prototype_no_prototype(self):
        config = {"prototype": None}
        with self.assertRaises(NameError) as context:
            _get_prototype(config)
        self.assertEqual(str(context.exception), "No prototype specified")

    def test_check_coherence_valid(self):
        prototype = ["features", "mitigation"]
        config = {
            "features": {"type": "PCA"},
            "mitigation": {"type": "CorrelationRemover"}
        }
        with self.assertRaises(Exception) as context:
            _check_coherence(prototype, config)
        self.assertEqual(str(context.exception), "PCA before CorrelationRemover")

    def test_check_coherence_invalid(self):
        prototype = ["mitigation", "features"]
        config = {
            "features": {"type": "PCA"},
            "mitigation": {"type": "CorrelationRemover"}
        }
        try:
            _check_coherence(prototype, config)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_prepare_indexes(self):
        categorical_indicator = [False, True, False, True]
        sensitive_indicator = [False, True, True, False]
        result = _prepare_indexes(categorical_indicator, sensitive_indicator)
        expected = {
            "num_features": [0, 2],
            "cat_features": [1, 3],
            "sen_num_features": [2],
            "sen_cat_features": [1]
        }
        self.assertEqual(result, expected)

    def test_prepare_parameters_mlpclassifier(self):
        config = {"step": {"type": "MLPClassifier", "n_neurons": 10, "n_hidden_layers": 2}}
        step = "step"
        indexes = None
        result = _prepare_parameters(config, step, indexes)
        expected = {"hidden_layer_sizes": 20}
        self.assertEqual(result, expected)

    def test_prepare_parameters_correlationremover(self):
        config = {"step": {"type": "CorrelationRemover"}}
        step = "step"
        indexes = {"sen_num_features": [0], "sen_cat_features": [1]}
        result = _prepare_parameters(config, step, indexes)
        expected = {"sensitive_feature_ids": [0, 1]}
        self.assertEqual(result, expected)


    def test_adjust_indexes_discretization(self):
        step = "discretization"
        config = {}
        indexes = {
            "num_features": [0, 1],
            "cat_features": [2, 3],
            "sen_num_features": [0],
            "sen_cat_features": [3]
        }
        p_pipeline = None
        result = _adjust_indexes(step, config, indexes, p_pipeline)
        expected = {
            "num_features": [],
            "cat_features": [0, 1, 2, 3],
            "sen_num_features": [],
            "sen_cat_features": [0, 3]
        }
        self.assertEqual(result, expected)

    # Additional tests for other steps and scenarios should be included similarly


class TestPrepareOperator(unittest.TestCase):
    
    def setUp(self):
        self.seed = 42
        self.indexes = {
            "num_features": [0, 1, 2],
            "cat_features": [3, 4],
            "sen_num_features": [0],
            "sen_cat_features": [3]
        }

    def test_random_forest_classifier(self):
        config = {
            "classification": {
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 3
            }
        }
        operator_parameters = {
            "n_estimators": 100,
            "max_depth": 3
        }
        step = "classification"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, RandomForestClassifier)
        self.assertEqual(result.get_params()["random_state"], 42)
        self.assertEqual(result.get_params()["n_estimators"], 100)
        self.assertEqual(result.get_params()["max_depth"], 3)

    def test_pca(self):
        config = {
            "features": {
                "type": "PCA",
                "n_components": 2
            }
        }
        operator_parameters = {
            "n_components": 2
        }
        step = "features"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, PCA)
        self.assertEqual(result.get_params()["n_components"], 2)

    def test_select_k_best(self):
        config = {
            "features": {
                "type": "SelectKBest",
                "k": 3
            }
        }
        operator_parameters = {
            "k": 3
        }
        step = "features"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, SelectKBest)
        self.assertEqual(result.get_params()["k"], 3)

    def test_simple_imputer(self):
        config = {
            "imputation": {
                "type": "SimpleImputer",
                "strategy": "mean"
            }
        }
        operator_parameters = {
            "strategy": "mean"
        }
        step = "imputation"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, SimpleImputer)
        self.assertEqual(result.get_params()["strategy"], "mean")

    def test_robust_scaler(self):
        config = {
            "normalization": {
                "type": "RobustScaler"
            }
        }
        operator_parameters = {}
        step = "normalization"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, ColumnTransformer)
        self.assertIsInstance(result.transformers[0][1].steps[0][1], RobustScaler)

    def test_near_miss(self):
        config = {
            "sampling": {
                "type": "NearMiss",
                "version": 1
            }
        }
        operator_parameters = {
            "version": 1
        }
        step = "sampling"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, NearMiss)
        self.assertEqual(result.get_params()["version"], 1)

    def test_smote(self):
        config = {
            "sampling": {
                "type": "SMOTE"
            }
        }
        operator_parameters = {}
        step = "sampling"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, SMOTE)

    def test_logistic_regression(self):
        config = {
            "classification": {
                "type": "LogisticRegression",
                "C": 1.0
            }
        }
        operator_parameters = {
            "C": 1.0
        }
        step = "classification"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, LogisticRegression)
        self.assertEqual(result.get_params()["C"], 1.0)

    def test_xgb_classifier(self):
        config = {
            "classification": {
                "type": "XGBClassifier",
                "n_estimators": 50
            }
        }
        operator_parameters = {
            "n_estimators": 50
        }
        step = "classification"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, XGBClassifier)
        self.assertEqual(result.get_params()["n_estimators"], 50)

    def test_correlation_remover(self):
        config = {
            "mitigation": {
                "type": "CorrelationRemover"
            }
        }
        operator_parameters = {
            "sensitive_feature_ids": [0, 3]
        }
        step = "mitigation"
        result = _prepare_operator(config, step, self.seed, self.indexes, operator_parameters)
        self.assertIsInstance(result, CorrelationRemover)
        self.assertEqual(result.get_params()["sensitive_feature_ids"], [0, 3])


class TestAdjustIndexes(unittest.TestCase):

    def setUp(self):
        self.indexes = {
            "num_features": [0, 1, 2],
            "cat_features": [3, 4],
            "sen_num_features": [0, 2],
            "sen_cat_features": [3]
        }

    def test_discretization(self):
        step = "discretization"
        config = {}
        p_pipeline = MagicMock()

        result = _adjust_indexes(step, config, self.indexes, p_pipeline)

        expected_indexes = {
            "num_features": [],
            "cat_features": [0, 1, 2, 3, 4],
            "sen_num_features": [],
            "sen_cat_features": [0, 2, 3]
        }
        self.assertEqual(result, expected_indexes)

    def test_encoding(self):
        step = "encoding"
        config = {}
        p_pipeline = MagicMock()

        result = _adjust_indexes(step, config, self.indexes, p_pipeline)

        expected_indexes = {
            "num_features": [0, 1, 2],
            "cat_features": [3, 4],
            "sen_num_features": [0, 2],
            "sen_cat_features": [3]
        }
        self.assertEqual(result, expected_indexes)

    def test_normalization(self):
        step = "normalization"
        config = {}
        p_pipeline = MagicMock()

        result = _adjust_indexes(step, config, self.indexes, p_pipeline)

        expected_indexes = {
            "num_features": [0, 1, 2],
            "cat_features": [3, 4],
            "sen_num_features": [0, 2],
            "sen_cat_features": [3]
        }
        self.assertEqual(result, expected_indexes)

    def test_features_pca(self):
        step = "features"
        config = {
            "features": {
                "type": "PCA",
                "n_components": 2
            }
        }
        p_pipeline = MagicMock()

        result = _adjust_indexes(step, config, self.indexes, p_pipeline)

        expected_indexes = {
            "num_features": [0, 1],
            "cat_features": [],
            "sen_num_features": [],
            "sen_cat_features": []
        }
        self.assertEqual(result, expected_indexes)

    def test_features_select_k_best(self):
        step = "features"
        config = {
            "features": {
                "type": "SelectKBest"
            }
        }
        p_pipeline = MagicMock()
        p_pipeline.return_value = [MagicMock()]
        p_pipeline.return_value[-1].get_support.return_value = [0, 2, 3]

        result = _adjust_indexes(step, config, self.indexes, p_pipeline)

        expected_indexes = {
            "num_features": [0, 1],
            "cat_features": [2],
            "sen_num_features": [0, 1],
            "sen_cat_features": [2]
        }
        self.assertEqual(result, expected_indexes)

    def test_mitigation(self):
        step = "mitigation"
        config = {}
        p_pipeline = MagicMock()

        result = _adjust_indexes(step, config, self.indexes, p_pipeline)

        expected_indexes = {
            "num_features": [0, 1],
            "cat_features": [],
            "sen_num_features": [],
            "sen_cat_features": []
        }
        self.assertEqual(result, expected_indexes)

if __name__ == "__main__":
    unittest.main()