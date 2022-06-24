#!/bin/bash
wget https://github.com/QueueInc/HAMLET/releases/download/$5/hamlet-$5-all.jar
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python automl/run_baseline.py --workspace $1 --metric $2 --mode $3 --batch_size $4 --version $5
deactivate