#!/bin/bash
python automl/main.py --dataset "iris" --metric "balanced_accuracy" --mode "max" --batch_size 25 --input_path "resources/automl/input/automl_input_4.json" --output_path "resources/automl/output/automl_output_4.json" --seed 42
