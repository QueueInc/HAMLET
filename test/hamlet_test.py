import unittest

from unittest.mock import MagicMock

# from automl.hamlet.engine import run

class HamletTest(unittest.TestCase):

    def test_main(self):
        args = MagicMock()
        args.seed = 42
        args.fair_metric = "demographic_parity"
        args.metric = "balanced_accuracy"
        args.input_path = "resources/automl_input_1.json"
        args.output_path = "resources/automl_output_1.json"
        args.dataset = "31"
        args.mode = "max"
        args.batch_size = 500
        args.time_budget = 60

        # run(args)

        
if __name__ == "__main__":
    unittest.main()