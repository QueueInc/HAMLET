import unittest
from unittest.mock import MagicMock

from context import hamlet
from hamlet.objective import _adjust_indexes

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