#!/bin/bash
python automl/main.py --dataset 29 --metric "balanced_accuracy" --mode "max" --batch_size 250 --input_path "resources_nn/automl/input/automl_input_3.json" --output_path "resources_nn/automl/output/automl_output_3.json" --seed 42
