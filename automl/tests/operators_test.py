import unittest

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from fairlearn.preprocessing import CorrelationRemover
from sklearn.compose import ColumnTransformer

from context import hamlet
from hamlet.objective import _prepare_operator


class TestPrepareOperator(unittest.TestCase):

    def setUp(self):
        self.seed = 42
        self.indexes = {
            "num_features": [0, 1, 2],
            "cat_features": [3, 4],
            "sen_num_features": [0],
            "sen_cat_features": [3],
            "feature_names": ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"],
        }

    def test_random_forest_classifier(self):
        config = {
            "classification": {
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 3,
            }
        }
        operator_parameters = {"n_estimators": 100, "max_depth": 3}
        step = "classification"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, RandomForestClassifier)
        self.assertEqual(result.get_params()["random_state"], 42)
        self.assertEqual(result.get_params()["n_estimators"], 100)
        self.assertEqual(result.get_params()["max_depth"], 3)

    def test_pca(self):
        config = {"features": {"type": "PCA", "n_components": 2}}
        operator_parameters = {"n_components": 2}
        step = "features"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, PCA)
        self.assertEqual(result.get_params()["n_components"], 2)

    def test_select_k_best(self):
        config = {"features": {"type": "SelectKBest", "k": 3}}
        operator_parameters = {"k": 3}
        step = "features"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, SelectKBest)
        self.assertEqual(result.get_params()["k"], 3)

    def test_simple_imputer(self):
        config = {"imputation": {"type": "SimpleImputer", "strategy": "mean"}}
        operator_parameters = {"strategy": "mean"}
        step = "imputation"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, SimpleImputer)
        self.assertEqual(result.get_params()["strategy"], "mean")

    def test_robust_scaler(self):
        config = {"normalization": {"type": "RobustScaler"}}
        operator_parameters = {}
        step = "normalization"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, ColumnTransformer)
        self.assertIsInstance(result.transformers[0][1].steps[0][1], RobustScaler)

    def test_near_miss(self):
        config = {"sampling": {"type": "NearMiss", "version": 1}}
        operator_parameters = {"version": 1}
        step = "sampling"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, NearMiss)
        self.assertEqual(result.get_params()["version"], 1)

    def test_smote(self):
        config = {"sampling": {"type": "SMOTE"}}
        operator_parameters = {}
        step = "sampling"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, SMOTE)

    def test_logistic_regression(self):
        config = {"classification": {"type": "LogisticRegression", "C": 1.0}}
        operator_parameters = {"C": 1.0}
        step = "classification"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, LogisticRegression)
        self.assertEqual(result.get_params()["C"], 1.0)

    def test_xgb_classifier(self):
        config = {"classification": {"type": "XGBClassifier", "n_estimators": 50}}
        operator_parameters = {"n_estimators": 50}
        step = "classification"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, XGBClassifier)
        self.assertEqual(result.get_params()["n_estimators"], 50)

    def test_correlation_remover(self):
        config = {"mitigation": {"type": "CorrelationRemover"}}
        operator_parameters = {"sensitive_feature_ids": [0, 3]}
        step = "mitigation"
        result = _prepare_operator(
            config, step, self.seed, self.indexes, operator_parameters
        )
        self.assertIsInstance(result, CorrelationRemover)
        self.assertEqual(result.get_params()["sensitive_feature_ids"], [0, 3])


if __name__ == "__main__":
    unittest.main()
