#!/bin/bash
docker stop hamlet_just_automl
docker rm hamlet_just_automl
docker build -t hamlet_just_automl .
docker run --name hamlet_just_automl --volume $(pwd):/home --detach -t hamlet_just_automl
docker exec hamlet_just_automl bash ./run_experiments.sh