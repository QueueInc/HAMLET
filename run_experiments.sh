#!/bin/bash
python automl/main.py --dataset "wine" --metric "accuracy" --mode "max" --batch_size 200 --input_path "resources/automl/input/automl_input_1.json" --output_path "resources/automl/output/automl_output_1.json" --seed 42
