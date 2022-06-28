#!/bin/bash
python automl/main.py --dataset 29 --metric "balanced_accuracy" --mode "max" --batch_size 5000 --input_path "temp/automl_input.json" --output_path "temp/automl_output.json" --seed 42
