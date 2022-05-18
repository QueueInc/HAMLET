#!/bin/bash

python automl/main.py \
    -dataset "iris" \
    -metric "balanced_accuracy" \
    -mode "max" \
    -batch_size 25 \
    -input_path "/workspaces/HAMLET/resources/automl_input.json" \
    -output_path "/workspaces/HAMLET/resources/automl_output.json" \
    -seed 42