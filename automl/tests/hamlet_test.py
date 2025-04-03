import unittest
from unittest.mock import MagicMock

from context import hamlet
from hamlet import run


class TestHamlet(unittest.TestCase):

    def test_balanced_accuracy(self):

        args = MagicMock()
        args.seed = 42
        args.fair_metric = "demographic_parity_difference"
        # args.sensitive_features = "8_12"
        args.metric = "balanced_accuracy"
        args.input_path = "automl/resources/new_automl_input_1.json"
        args.output_path = "automl/resources/new_automl_output_1.json"
        args.dataset = "31"
        args.mode = "max"
        args.batch_size = 25
        args.time_budget = 60

        best, _ = run(args)

        self.assertGreater(len(best), 0)

    def test_equalized_odds(self):

        args = MagicMock()
        args.seed = 42
        args.fair_metric = "equalized_odds_ratio"
        # args.sensitive_features = "8_12"
        args.metric = "balanced_accuracy"
        args.input_path = "automl/resources/new_automl_input_1.json"
        args.output_path = "automl/resources/new_automl_output_1.json"
        args.dataset = "31"
        args.mode = "max"
        args.batch_size = 25
        args.time_budget = 60

        best, _ = run(args)

        self.assertGreater(len(best), 0)


if __name__ == "__main__":
    unittest.main()
