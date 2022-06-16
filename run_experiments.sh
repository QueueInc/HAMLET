#!/bin/bash
python automl/main.py --dataset 29 --metric "balanced_accuracy" --mode "max" --batch_size 250 --input_path "resources/automl/input/automl_input_1.json" --output_path "resources/automl/output/automl_output_1.json" --seed 42
