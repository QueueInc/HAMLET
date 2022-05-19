#!/bin/bash

python automl/main.py \
    -dataset "iris" \
    -metric "balanced_accuracy" \
    -mode "max" \
    -batch_size 25 \
    -input_path "/workspaces/HAMLET/resources/automl/input/automl_input_10.json" \
    -output_path "/workspaces/HAMLET/resources/automl/output/automl_output_10.json" \
    -seed 42