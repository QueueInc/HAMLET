#!/bin/bash
python automl/main.py --dataset wine --metric "balanced_accuracy" --mode "max" --batch_size 0 --time_budget 60 --input_path "temp/automl_input.json" --output_path "temp/automl_output.json" --seed 42
